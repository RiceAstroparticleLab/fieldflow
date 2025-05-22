"""Training infrastructure for FieldFlow continuous normalizing flow models.

This module provides loss functions, training loops, and utilities for training
CNF models to learn drift fields from position reconstruction data.
"""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.interpolate import RegularGridInterpolator
from jaxtyping import Array, PRNGKeyArray, PyTree
from tqdm import trange

from fieldflow.posrec import generate_samples_for_cnf
from fieldflow.utils import compute_r

if TYPE_CHECKING:
    from fieldflow.config import Config


def rolloff_func(x: Array, rolloff: float = 1e-2) -> Array:
    """Apply rolloff regularization to prevent numerical issues.

    This function ensures that probabilities don't get too close to zero,
    which can cause numerical instability in log computations.

    Args:
        x: Input array to regularize
        rolloff: Minimum value parameter

    Returns:
        Regularized array
    """
    return x + rolloff * jnp.exp(-x / rolloff)


def curl_loss(
    key: PRNGKeyArray,  # noqa: ARG001
    model: eqx.Module,
    z: float,
    x: Array,
    extract_max_z: float = 10.0,  # noqa: ARG001
) -> float:
    """Compute curl penalty for vector field to encourage curl-free flow.

    This loss encourages the learned drift field to have minimal curl,
    which is a physical constraint for certain types of fields.

    Args:
        key: Random key for sampling (unused but kept for interface)
        model: CNF model with func_drift method
        z: Current z coordinate (time parameter)
        x: Spatial coordinates [x, y]
        extract_max_z: Maximum z value for random sampling (unused)

    Returns:
        Curl penalty loss value
    """
    jac_drift = jax.jacfwd(lambda a: model.func_drift(z, a))(x)

    # For 2D vector field, curl is ∂fy/∂x - ∂fx/∂y
    curl_penalty = (jac_drift[1, 0] - jac_drift[0, 1])**2

    return curl_penalty


def single_likelihood_loss(
    key: PRNGKeyArray,
    model: eqx.Module,
    condition: Array,
    t1: float,
    z: float,
    posrec_model: eqx.Module,
    civ_map: RegularGridInterpolator,
    tpc_r: float,
    radius_buffer: float = 20.0,
    min_p: float = 1e-3,
    n_samples: int = 4,
    curl_loss_multiplier: float = 1000.0,
) -> float:
    """Compute likelihood loss for a single data point.

    This function computes the negative log-likelihood for a single event,
    incorporating survival probability from CIV maps and curl penalty.

    Args:
        key: Random key for sampling
        model: CNF model to train
        condition: Hit pattern conditioning information
        t1: Target time for transformation (scaled z coordinate)
        z: Physical z coordinate
        posrec_model: Pretrained position reconstruction model
        civ_map: Charge-in-volume survival probability map
        tpc_r: TPC radius for boundary constraints
        radius_buffer: Buffer for predictions beyond TPC radius
        min_p: Minimum survival probability (for numerical stability)
        n_samples: Number of samples for Monte Carlo estimation
        curl_loss_multiplier: Weight for curl penalty term

    Returns:
        Negative log-likelihood loss value
    """
    keys = jax.random.split(key, 2)

    # Generate samples from position reconstruction flow
    samples = generate_samples_for_cnf(
        keys[0], condition[jnp.newaxis, ...], n_samples, posrec_model,
        tpc_r, radius_buffer
    )

    # Transform samples through CNF model
    transformed_samples, logdet = eqx.filter_vmap(
        lambda y: model.transform_and_log_det(y=y, t1=t1)
    )(samples)

    # Compute radii of transformed samples
    sample_r = compute_r(transformed_samples)

    # Compute survival probabilities from CIV map
    civ_coords = jnp.vstack((sample_r, jnp.repeat(z, n_samples))).T
    vec_civ_map = jax.vmap(civ_map)
    p_surv = vec_civ_map(civ_coords)

    # Apply rolloff regularization and boundary constraints
    p_surv = rolloff_func(p_surv, min_p) * jnp.prod(
        jnp.where(
            sample_r <= tpc_r,
            jnp.ones_like(sample_r),
            jnp.exp((tpc_r - sample_r) / 10000)
        )
    )

    # Compute negative log-likelihood using logsumexp for numerical stability
    likelihood_loss_val = (
        -jax.nn.logsumexp(a=logdet, b=p_surv) + jnp.log(n_samples)
    )

    # Add curl penalty
    curl_penalty = curl_loss_multiplier * curl_loss(
        keys[1], model, t1, transformed_samples[0]
    )

    return likelihood_loss_val + curl_penalty


@eqx.filter_jit
def likelihood_loss(
    model: eqx.Module,
    key: PRNGKeyArray,
    conditions: Array,
    t1s: Array,
    zs: Array,
    posrec_model: eqx.Module,
    civ_map: RegularGridInterpolator,
    tpc_r: float,
    n_samples: int = 4,
    **kwargs
) -> float:
    """Compute vectorized likelihood loss over a batch of data.

    Args:
        model: CNF model to train
        key: Random key for sampling
        conditions: Batch of hit pattern conditions
        t1s: Batch of scaled z coordinates
        zs: Batch of physical z coordinates
        posrec_model: Pretrained position reconstruction model
        civ_map: Charge-in-volume survival probability map
        tpc_r: TPC radius for boundary constraints
        n_samples: Number of samples per data point
        **kwargs: Additional arguments passed to single_likelihood_loss

    Returns:
        Mean loss over the batch
    """
    keys = jax.random.split(key, len(zs))
    vec_loss = eqx.filter_vmap(
        lambda k, cond, t1, z: single_likelihood_loss(
            k, model, cond, t1, z, posrec_model, civ_map, tpc_r,
            n_samples=n_samples, **kwargs
        )
    )
    return jnp.mean(vec_loss(keys, conditions, t1s, zs))


def create_optimizer(config: "Config") -> optax.GradientTransformation:
    """Create optimizer from configuration.

    Args:
        config: Configuration object containing training parameters

    Returns:
        Configured optax optimizer
    """
    # Create learning rate schedule
    optax_sched = optax.join_schedules([
        optax.constant_schedule(config.training.learning_rate),
        optax.constant_schedule(config.training.learning_rate * 0.1),
        optax.constant_schedule(config.training.learning_rate * 0.01)
    ], [25, 30])

    # Create base optimizer
    optimizer = optax.adamw(
        learning_rate=optax_sched,
        weight_decay=config.training.weight_decay
    )

    # Add gradient clipping and finite check with configurable steps
    optimizer = optax.apply_if_finite(
        optax.MultiSteps(
            optimizer,
            every_k_schedule=config.training.multisteps_every_k
        ),
        max_consecutive_errors=4
    )

    return optimizer


def train(
    key: PRNGKeyArray,
    model: eqx.Module,
    optim: optax.GradientTransformation,
    epochs: int,
    conditions: Array,
    t1s: Array,
    zs: Array,
    posrec_model: eqx.Module,
    civ_map: RegularGridInterpolator,
    n_train: int,
    n_batch: int,
    n_samples: int,
    n_test: int,
    tpc_r: float,
    radius_buffer: float = 20.0,
    use_best: bool = False,
    loss_fn: Callable = likelihood_loss,
    num_devices: int = 1,
) -> tuple[eqx.Module, list, list]:
    """Train a continuous normalizing flow model.

    This function implements the main training loop with batching, progress
    tracking, and optional best model selection based on validation loss.

    Args:
        key: Random key for training
        model: CNF model to train
        optim: Optimizer (from create_optimizer)
        epochs: Number of training epochs
        conditions: All hit pattern conditions
        t1s: All scaled z coordinates
        zs: All physical z coordinates
        posrec_model: Pretrained position reconstruction model
        civ_map: Charge-in-volume survival probability map
        n_train: Number of training samples
        n_batch: Batch size
        n_samples: Samples per likelihood evaluation
        n_test: Number of test samples
        tpc_r: TPC radius for boundary constraints
        radius_buffer: Buffer for predictions beyond TPC radius
        use_best: Whether to return best model based on validation loss
        loss_fn: Loss function to use
        num_devices: Number of devices to use for data parallelization

    Returns:
        Tuple of (trained_model, train_loss_history, test_loss_history)
    """
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Set up device mesh for multi-GPU training
    devices = jax.devices()[:num_devices]
    device_mesh = jax.sharding.Mesh(devices, ("data",))
    
    # Create data sharding specification
    data_sharding = jax.sharding.NamedSharding(
        device_mesh, jax.sharding.PartitionSpec("data")
    )
    
    # Validate batch size and warn if not evenly divisible
    if n_batch % num_devices != 0:
        warnings.warn(
            f"Batch size {n_batch} is not evenly divisible by num_devices "
            f"{num_devices}. Some devices may process fewer samples per batch, "
            f"which could lead to slightly uneven GPU utilization.",
            UserWarning,
            stacklevel=2
        )

    # Split data into train/test and convert training data to NumPy arrays
    # This keeps the full dataset in CPU memory instead of GPU memory
    cond_train_orig_np = np.asarray(conditions[:-n_test])
    t1s_train_orig_np = np.asarray(t1s[:-n_test])
    zs_train_orig_np = np.asarray(zs[:-n_test])

    # Test data stays as JAX arrays since it's small
    cond_test = conditions[-n_test:]
    t1s_test = t1s[-n_test:]
    zs_test = zs[-n_test:]

    n_data_loops = n_train // n_batch

    @eqx.filter_jit(donate="all")
    def make_step(
        model: eqx.Module,
        opt_state: PyTree,
        key: PRNGKeyArray,
        conds: Array,
        t1s: Array,
        zs: Array,
        sharding: jax.sharding.Sharding
    ) -> tuple[eqx.Module, PyTree, float]:
        """Single training step with multi-device support."""
        # Generate replicated sharding from data sharding
        replicated = sharding.replicate()
        
        # Shard model and opt_state (replicated across all devices)
        model_replicated, opt_state_replicated = eqx.filter_shard(
            (model, opt_state), replicated
        )
        
        # Shard data across devices  
        conds_sharded, t1s_sharded, zs_sharded = eqx.filter_shard(
            (conds, t1s, zs), sharding
        )
        
        # Compute loss and gradients (distributed across devices)
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
            model_replicated, key, conds_sharded, t1s_sharded, zs_sharded, 
            posrec_model, civ_map, tpc_r, n_samples=n_samples, 
            radius_buffer=radius_buffer
        )
        
        # Update model (gradients automatically aggregated across devices)
        updates, opt_state_new = optim.update(
            grads, opt_state_replicated, eqx.filter(model_replicated, eqx.is_array)
        )
        model_new = eqx.apply_updates(model_replicated, updates)
        
        # Return updated model and optimizer state
        return model_new, opt_state_new, loss_value

    # Training loop
    loop = trange(epochs)
    best_model = model
    train_loss_list = []
    test_loss_list = [
        loss_fn(model, key, cond_test, t1s_test, zs_test,
               posrec_model, civ_map, tpc_r, n_samples=n_samples,
               radius_buffer=radius_buffer)
    ]

    for _ in loop:
        key, thiskey = jax.random.split(key, 2)

        # Generate permutation indices using JAX (deterministic)
        indices_jax = jax.random.permutation(thiskey, jnp.arange(n_train))
        # Convert to NumPy for CPU-based indexing
        indices_np = np.array(indices_jax)
        
        # Use NumPy arrays for all data operations (stays in CPU)
        cond_train_np = cond_train_orig_np[indices_np]
        t1s_train_np = t1s_train_orig_np[indices_np]
        zs_train_np = zs_train_orig_np[indices_np]

        # Training steps for this epoch
        for j in range(n_data_loops):
            key, thiskey = jax.random.split(key, 2)
            
            # Extract batch as NumPy arrays (stays in CPU)
            batch_start = j * n_batch
            batch_end = (j + 1) * n_batch
            batch_conds_np = cond_train_np[batch_start:batch_end]
            batch_t1s_np = t1s_train_np[batch_start:batch_end]
            batch_zs_np = zs_train_np[batch_start:batch_end]
            
            # Convert numpy to JAX arrays (let make_step handle sharding)
            batch_conds = jnp.array(batch_conds_np)
            batch_t1s = jnp.array(batch_t1s_np)
            batch_zs = jnp.array(batch_zs_np)

            model, opt_state, train_loss = make_step(
                model, opt_state, thiskey, batch_conds, batch_t1s, batch_zs, data_sharding
            )
            train_loss_list.append(train_loss)

            # Update progress bar
            train_ma = jnp.mean(jnp.array(train_loss_list[-64:]))
            loop.set_postfix({
                "loss": f"{train_loss_list[-1]:0.2f}",
                "loss MA": f"{train_ma:0.3f}",
                "test loss": f"{test_loss_list[-1]:0.3f}",
            })

        # Evaluate on test set
        test_loss = loss_fn(
            model, key, cond_test, t1s_test, zs_test,
            posrec_model, civ_map, tpc_r, n_samples=n_samples,
            radius_buffer=radius_buffer
        )
        test_loss_list.append(test_loss)

        # Track best model
        if jnp.argmin(jnp.array(test_loss_list)) == len(test_loss_list) - 1:
            best_model = model

    if use_best:
        model = best_model

    return model, train_loss_list, test_loss_list


def train_model_from_config(
    key: PRNGKeyArray,
    model: eqx.Module,
    conditions: Array,
    t1s: Array,
    zs: Array,
    posrec_model: eqx.Module,
    civ_map: RegularGridInterpolator,
    config: "Config",
) -> tuple[eqx.Module, list, list]:
    """Train model using configuration parameters.

    Convenience function that creates optimizer and calls train() with
    parameters from the configuration object.

    Args:
        key: Random key for training
        model: CNF model to train
        conditions: Hit pattern conditions
        t1s: Scaled z coordinates
        zs: Physical z coordinates
        posrec_model: Pretrained position reconstruction model
        civ_map: Charge-in-volume survival probability map
        config: Configuration object

    Returns:
        Tuple of (trained_model, train_loss_history, test_loss_history)
    """
    optimizer = create_optimizer(config)

    return train(
        key=key,
        model=model,
        optim=optimizer,
        epochs=config.training.epochs,
        conditions=conditions,
        t1s=t1s,
        zs=zs,
        posrec_model=posrec_model,
        civ_map=civ_map,
        n_train=config.training.n_train,
        n_batch=config.training.batch_size,
        n_samples=config.training.n_samples,
        n_test=config.training.n_test,
        tpc_r=config.experiment.tpc_r,
        radius_buffer=config.posrec.radius_buffer,
        use_best=config.training.use_best,
        num_devices=config.training.num_devices,
    )
