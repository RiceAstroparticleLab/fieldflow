"""Training infrastructure for FieldFlow CNF models.

This module provides the loss functions and training loop for learning
electric field distortions in dual-phase TPCs. Training uses position
samples from a pretrained reconstruction flow weighted by
charge-insensitive-volume survival probabilities.
"""

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.scipy.interpolate import RegularGridInterpolator
from jaxtyping import Array, PRNGKeyArray, PyTree
from tqdm import trange

from fieldflow.posrec import generate_samples_for_cnf
from fieldflow.utils import compute_r

if TYPE_CHECKING:
    from fieldflow.config import Config


def rolloff_func(x: Array, rolloff: float = 1e-2) -> Array:
    """Apply soft lower bound to prevent log(0) numerical issues.

    Smoothly regularizes small values using: x + rolloff * exp(-x/rolloff).
    This approaches x for large values and rolloff for x near 0.

    Args:
        x: Input array (typically survival probabilities).
        rolloff: Soft minimum value parameter.

    Returns:
        Regularized array with values bounded away from zero.
    """
    return x + rolloff * jnp.exp(-x / rolloff)


def curl_loss(
    key: PRNGKeyArray,  # noqa: ARG001
    model: eqx.Module,
    z: float,
    x: Array,
    extract_max_z: float = 10.0,  # noqa: ARG001
) -> float:
    """Compute curl penalty for enforcing Maxwell's equations.

    For electrostatic fields, curl(E) = 0. This loss penalizes non-zero curl
    in the learned drift field when using the vector field approach (MLPFunc).
    Not needed when using scalar potential (DriftFromPotential) since gradient
    fields are curl-free by construction.

    Args:
        key: Unused (kept for interface compatibility).
        model: CNF model containing func_drift.
        z: Current z coordinate (ODE time parameter).
        x: Spatial coordinates (x, y) at which to evaluate curl.
        extract_max_z: Unused (kept for interface compatibility).

    Returns:
        Squared curl value (∂v_y/∂x - ∂v_x/∂y)².
    """
    jac_drift = jax.jacfwd(lambda a: model.func_drift(z, a, 0.0))(x)

    # For 2D vector field, curl is ∂fy/∂x - ∂fx/∂y
    curl_penalty = (jac_drift[1, 0] - jac_drift[0, 1]) ** 2

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
    scalar: bool = False,
) -> float:
    """Compute negative log-likelihood loss for a single event.

    The loss combines:
    1. Monte Carlo estimation of the likelihood using samples from the
       position reconstruction flow, transformed through the CNF
    2. CIV survival probability weighting
    3. Optional curl penalty for vector field approach

    Args:
        key: JAX random key for sampling.
        model: CNF model being trained.
        condition: Hit pattern conditioning vector.
        t1: ODE integration time (scaled z coordinate).
        z: Physical z coordinate in cm (for CIV lookup).
        posrec_model: Pretrained position reconstruction flow.
        civ_map: Charge-insensitive-volume survival probability interpolator.
        tpc_r: TPC radius in cm for boundary constraints.
        radius_buffer: Buffer beyond TPC radius for position sampling.
        min_p: Minimum survival probability (numerical stability).
        n_samples: Number of Monte Carlo samples.
        curl_loss_multiplier: Weight for curl penalty term.
        scalar: If True, skip curl penalty (using scalar potential method).

    Returns:
        Combined negative log-likelihood and curl penalty loss.
    """
    keys = jax.random.split(key, 2 + n_samples)

    # Generate samples from position reconstruction flow
    samples = generate_samples_for_cnf(
        keys[0],
        condition[jnp.newaxis, ...],
        n_samples,
        posrec_model,
        tpc_r,
        radius_buffer,
    )

    # Transform samples through CNF model
    transformed_samples, logdet = eqx.filter_vmap(
        lambda y, k: model.transform_and_log_det(y=y, t1=t1, key=k)
    )(samples, keys[1 : 1 + n_samples])

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
            jnp.exp((tpc_r - sample_r) / 100),
        )
    )

    # Compute negative log-likelihood using logsumexp for numerical stability
    likelihood_loss_val = -jax.nn.logsumexp(a=logdet, b=p_surv) + jnp.log(
        n_samples
    )

    if scalar:
        curl_penalty = 0
    else:
        # Add curl penalty
        curl_penalty = curl_loss_multiplier * curl_loss(
            keys[1 + n_samples], model, t1, transformed_samples[0]
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
    scalar = False,
    **kwargs,
) -> float:
    """Compute batched likelihood loss over multiple events.

    Vectorizes single_likelihood_loss over a batch of events and returns
    the mean loss.

    Args:
        model: CNF model being trained.
        key: JAX random key for sampling.
        conditions: Batch of hit patterns, shape (batch_size, cond_dim).
        t1s: Batch of ODE times (scaled z), shape (batch_size,).
        zs: Batch of physical z coordinates, shape (batch_size,).
        posrec_model: Pretrained position reconstruction flow.
        civ_map: CIV survival probability interpolator.
        tpc_r: TPC radius in cm.
        n_samples: Monte Carlo samples per event.
        scalar: If True, skip curl penalty.
        **kwargs: Additional arguments for single_likelihood_loss.

    Returns:
        Mean loss over the batch.
    """
    keys = jax.random.split(key, len(zs))
    vec_loss = eqx.filter_vmap(
            lambda k, cond, t1, z: single_likelihood_loss(
                k,
                model,
                cond,
                t1,
                z,
                posrec_model,
                civ_map,
                tpc_r,
                n_samples=n_samples,
                scalar = scalar,
                **kwargs,
            )
        )

    return jnp.mean(vec_loss(keys, conditions, t1s, zs))


def create_optimizer(config: "Config") -> optax.GradientTransformation:
    """Create AdamW optimizer with optional learning rate schedule.

    When enable_scheduler is True, uses a piecewise constant schedule:
    - Epochs 0-19: learning_rate
    - Epochs 20-69: learning_rate * 0.5
    - Epochs 70+: learning_rate * 0.1

    Args:
        config: Configuration with training.learning_rate,
            training.weight_decay, training.enable_scheduler, and
            training.multisteps_every_k.

    Returns:
        Configured optax optimizer with gradient accumulation and
        finite gradient checking.
    """
    # Create learning rate schedule
    if config.training.enable_scheduler:

        optax_sched = optax.join_schedules(
            [
                optax.constant_schedule(config.training.learning_rate),
                optax.constant_schedule(config.training.learning_rate * 0.5),
                optax.constant_schedule(config.training.learning_rate * 0.1),
            ],
            [20, 70, 150],
        )
    else:
        # Use constant learning rate at inputted value
        optax_sched = optax.constant_schedule(
            config.training.learning_rate
        )

    # Create base optimizer
    optimizer = optax.adamw(
        learning_rate=optax_sched, weight_decay=config.training.weight_decay
    )

    # Add gradient clipping and finite check with configurable steps
    optimizer = optax.apply_if_finite(
        optax.MultiSteps(
            optimizer, every_k_schedule=config.training.multisteps_every_k
        ),
        max_consecutive_errors=4,
    )

    return optimizer


def save_model(model, path):
    """Save model weights to disk using equinox serialization.

    Args:
        model: Equinox model to save.
        path: File path for saved model (typically .eqx extension).
    """
    eqx.tree_serialise_leaves(path, model)
    print(f"Model saved to {path}")


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
    save_iter: int = 1,
    save_file_name: str = "model",
    output_path: str = "",
    loss_fn: Callable = likelihood_loss,
    num_devices: int = 1,
    scalar: bool = False,
    epoch_start: int = 0,
) -> tuple[eqx.Module, list, list, int]:
    """Train a CNF model with multi-GPU support.

    Implements the main training loop with automatic data sharding across
    multiple GPUs. Data is resharded once per epoch for optimal performance.
    Saves periodic checkpoints and tracks train/validation loss history.

    Args:
        key: JAX random key for training.
        model: CNF model to train.
        optim: Optax optimizer (from create_optimizer).
        epochs: Number of training epochs.
        conditions: Full dataset of hit patterns.
        t1s: Full dataset of scaled z coordinates (ODE times).
        zs: Full dataset of physical z coordinates.
        posrec_model: Frozen pretrained position reconstruction model.
        civ_map: CIV survival probability interpolator.
        n_train: Number of samples for training split.
        n_batch: Batch size per training step.
        n_samples: Monte Carlo samples per event in loss computation.
        n_test: Number of samples for validation split.
        tpc_r: TPC radius in cm.
        radius_buffer: Buffer beyond TPC radius for position sampling.
        use_best: If True, return model with lowest validation loss.
        save_iter: Save checkpoint every N epochs.
        save_file_name: Base filename for checkpoints.
        output_path: Directory for saving checkpoints and loss logs.
        loss_fn: Loss function (default: likelihood_loss).
        num_devices: Number of GPUs for data parallelization.
        scalar: If True, use scalar potential (no curl penalty).
        epoch_start: Starting epoch number (for resuming training).

    Returns:
        Tuple of (model, train_losses, val_losses, best_epoch) where
        model is trained (or best if use_best=True).
    """
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Set up device mesh for multi-GPU training
    devices = jax.devices()[:num_devices]
    device_mesh = jax.sharding.Mesh(devices, ("batch",))

    # Create batch sharding specification (batches distributed across devices)
    batch_sharding = jax.sharding.NamedSharding(
        device_mesh, jax.sharding.PartitionSpec(None, "batch")
    )

    # Create replicated sharding specification
    replicated_sharding = jax.sharding.NamedSharding(
        device_mesh, jax.sharding.PartitionSpec()
    )

    # Calculate number of batches (may drop some samples for even distribution)
    n_batches = n_train // n_batch
    n_usable_samples = n_batches * n_batch

    # Validate batch distribution and warn if needed
    if n_batch % num_devices != 0:
        warnings.warn(
            f"Batch size {n_batch} is not evenly divisible by num_devices "
            f"{num_devices}. Some devices may process fewer samples per "
            f"batch, which could lead to slightly uneven GPU utilization.",
            UserWarning,
            stacklevel=2,
        )

    if n_usable_samples < n_train:
        warnings.warn(
            f"Dropping {n_train - n_usable_samples} samples to ensure even "
            f"batch distribution ({n_usable_samples} samples used).",
            UserWarning,
            stacklevel=2,
        )

    # Prepare training data (keep in JAX format)
    cond_train = conditions[:-n_test]
    t1s_train = t1s[:-n_test]
    zs_train = zs[:-n_test]

    # Test data - shard once as a single large batch across devices
    cond_test = conditions[-n_test:]
    t1s_test = t1s[-n_test:]
    zs_test = zs[-n_test:]

    # Reshape test data to batch-first format: (1, n_test, ...)
    cond_test_batched = cond_test.reshape(1, n_test, -1)
    t1s_test_batched = t1s_test.reshape(1, n_test)
    zs_test_batched = zs_test.reshape(1, n_test)

    cond_test_sharded = jax.device_put(cond_test_batched, batch_sharding)
    t1s_test_sharded = jax.device_put(t1s_test_batched, batch_sharding)
    zs_test_sharded = jax.device_put(zs_test_batched, batch_sharding)

    # Shard model, optimizer state, and frozen posrec_model once (replicated)
    model_sharded = eqx.filter_shard(model, replicated_sharding)
    opt_state_sharded = eqx.filter_shard(opt_state, replicated_sharding)
    posrec_model_sharded = eqx.filter_shard(posrec_model, replicated_sharding)

    @eqx.filter_jit
    def make_step(
        model: eqx.Module,
        opt_state: PyTree,
        key: PRNGKeyArray,
        batch_data: tuple[Array, Array, Array],
        posrec_model: eqx.Module,
        scalar: bool
    ) -> tuple[eqx.Module, PyTree, float]:
        """Single training step with pre-sharded batch data."""
        # Data and models are already sharded - extract batch components
        batch_conds, batch_t1s, batch_zs = batch_data

        # Compute loss and gradients (distributed across devices)
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
            model,
            key,
            batch_conds,
            batch_t1s,
            batch_zs,
            posrec_model,
            civ_map,
            tpc_r,
            n_samples=n_samples,
            radius_buffer=radius_buffer,
            scalar=scalar,
        )

        # Update model (gradients automatically aggregated across devices)
        updates, opt_state_new = optim.update(
            grads,
            opt_state,
            eqx.filter(model, eqx.is_array),
        )
        model_new = eqx.apply_updates(model, updates)

        return model_new, opt_state_new, loss_value

    # Training loop
    loop = trange(epochs)
    best_model = model
    average_train_loss_list = []
    train_loss_list = []
    test_loss_list = [
        loss_fn(
            model_sharded,
            key,
            cond_test_sharded[0],
            t1s_test_sharded[0],
            zs_test_sharded[0],
            posrec_model_sharded,
            civ_map,
            tpc_r,
            n_samples=n_samples,
            radius_buffer=radius_buffer,
            scalar = scalar,
        )
    ]

    output_path = Path(output_path)
    with open(str(output_path / "test_losses.json"), "a") as f:
        f.write(json.dumps(float(test_loss_list[-1])) + "\n")

    best_epoch = 0
    for epoch in loop:

        epoch_shift = epoch + epoch_start
        key, thiskey = jax.random.split(key, 2)

        # Get data for this epoch (each epoch goes through n_usable_samples)
        total_patterns = len(cond_train) - (len(cond_train)%n_usable_samples)
        index_low = epoch_shift*n_usable_samples % total_patterns
        index_high = (epoch_shift+1)*n_usable_samples % total_patterns
        if index_high == 0:
            index_high = total_patterns
        shuffled_conds = cond_train[index_low:index_high]
        shuffled_t1s = t1s_train[index_low:index_high]
        shuffled_zs = zs_train[index_low:index_high]

        # Reshape to batch-first format: (n_batches, batch_size, ...)
        epoch_conds = shuffled_conds[:n_usable_samples].reshape(
            n_batches, n_batch, -1
        )
        epoch_t1s = shuffled_t1s[:n_usable_samples].reshape(n_batches, n_batch)
        epoch_zs = shuffled_zs[:n_usable_samples].reshape(n_batches, n_batch)

        # Shard data across devices once per epoch
        # Each device gets a subset of batches
        sharded_conds = jax.device_put(epoch_conds, batch_sharding)
        sharded_t1s = jax.device_put(epoch_t1s, batch_sharding)
        sharded_zs = jax.device_put(epoch_zs, batch_sharding)

        # Training steps for this epoch - simple batch indexing
        for j in range(n_batches):
            key, thiskey = jax.random.split(key, 2)

            # Extract batch - data is already sharded optimally
            batch_data = (sharded_conds[j], sharded_t1s[j], sharded_zs[j])

            model_sharded, opt_state_sharded, train_loss = make_step(
                model_sharded,
                opt_state_sharded,
                thiskey,
                batch_data,
                posrec_model_sharded,
                scalar = scalar,
            )
            train_loss_list.append(train_loss)
            with open(str(output_path / "train_losses.json"), "a") as f:
                f.write(json.dumps(float(train_loss_list[-1])) + "\n")

            # Update progress bar
            train_ma = jnp.mean(jnp.array(train_loss_list[-64:]))
            loop.set_postfix(
                {
                    "loss": f"{train_loss_list[-1]:0.2f}",
                    "loss MA": f"{train_ma:0.3f}",
                    "validation loss": f"{test_loss_list[-1]:0.3f}",
                }
            )

        # Update unsharded model for best model tracking
        model = eqx.filter_shard(model_sharded, replicated_sharding)

        # Evaluate on test set using unsharded model and test data
        test_loss = loss_fn(
            model,
            key,
            cond_test_sharded[0],
            t1s_test_sharded[0],
            zs_test_sharded[0],
            posrec_model_sharded,
            civ_map,
            tpc_r,
            n_samples=n_samples,
            radius_buffer=radius_buffer,
            scalar=scalar,
        )
        test_loss_list.append(test_loss)
        average_train_loss_list.append(jnp.nanmean(jnp.array(train_loss_list[-n_batches:])))

        with open(str(output_path / "val_losses.json"), "a") as f:
            f.write(json.dumps(float(test_loss_list[-1])) + "\n")
        with open(str(output_path / "average_train_losses.json"), "a") as f:
            f.write(json.dumps(float(average_train_loss_list[-1])) + "\n")

        # Track best model
        if jnp.argmin(jnp.array(test_loss_list)) == len(test_loss_list) - 1:
            best_model = model
            best_epoch = epoch

        if epoch % save_iter == 0:
            save_model(
                model, str(Path(output_path) / f"{save_file_name}_{epoch}.eqx")
            )

    if use_best:
        model = best_model

    return model, train_loss_list, test_loss_list, best_epoch


def train_model_from_config(
    key: PRNGKeyArray,
    model: eqx.Module,
    conditions: Array,
    t1s: Array,
    zs: Array,
    posrec_model: eqx.Module,
    civ_map: RegularGridInterpolator,
    config: "Config",
    output_path: str = "",
) -> tuple[eqx.Module, list, list, int]:
    """Train a CNF model using parameters from a Config object.

    Convenience wrapper that extracts training parameters from the config
    and calls train(). Creates the optimizer internally.

    Args:
        key: JAX random key for training.
        model: CNF model to train.
        conditions: Full dataset of hit patterns.
        t1s: Full dataset of scaled z coordinates.
        zs: Full dataset of physical z coordinates.
        posrec_model: Pretrained position reconstruction model.
        civ_map: CIV survival probability interpolator.
        config: Configuration object with all training parameters.
        output_path: Directory for saving checkpoints.

    Returns:
        Tuple of (model, train_losses, val_losses, best_epoch).
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
        save_iter=config.training.save_iter,
        save_file_name=config.training.save_file_name,
        output_path=output_path,
        num_devices=config.training.num_devices,
        scalar=config.model.scalar,
        epoch_start=config.training.epoch_start,
    )
