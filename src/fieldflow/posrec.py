"""Pretrained conditional normalizing flow for position reconstruction from
detector hit patterns, including coordinate transformations.
"""

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections import (
    Affine,
    Chain,
    Invert,
    RationalQuadraticSpline,
    Tanh,
)
from flowjax.distributions import StandardNormal
from flowjax.flows import coupling_flow
from jaxtyping import Array, PRNGKeyArray

if TYPE_CHECKING:
    from config import Config

# Transformation parameters from neural_net_defs.py
EPS = 1e-7  # Avoid potential numerical issues
SCALE = 1.0  # Scaling for tanh, 1 by default

# Coordinate transformation chain
unconstrain_transform = Chain(
    [
        Affine(
            loc=-jnp.ones(2) + EPS, scale=(1 - EPS) * jnp.array([2.0, 2.0])
        ),  # [-1+eps, 1-eps]
        Invert(Tanh(shape=(2,))),  # arctanh (to unbounded)
        Affine(loc=jnp.zeros(2) + EPS, scale=jnp.array([SCALE, SCALE])),
    ]
)

# Vectorized transformation functions
constrain_vec = jax.vmap(unconstrain_transform.inverse)


@jax.jit
def data_inv_transformation(
    data: Array, tpc_r: float, radius_buffer: float
) -> Array:
    """Transform flow coordinates back to physical (x,y) coordinates.

    This function applies the full coordinate transformation chain to convert
    from normalized flow space back to physical detector coordinates.

    Args:
        data: Array of shape (N, 2) in flow coordinate space
        tpc_r: TPC radius in cm
        radius_buffer: Buffer for predictions beyond TPC radius

    Returns:
        Array of shape (N, 2) in physical coordinates (cm)
    """
    max_pred = tpc_r + radius_buffer
    # Apply the constraint transformation (includes tanh)
    constrained_data = constrain_vec(data)

    # Convert from [0,1] space to physical coordinates
    data_0 = (constrained_data[:, 0] - 0.5) * max_pred * 2
    data_1 = (constrained_data[:, 1] - 0.5) * max_pred * 2
    return jnp.stack([data_0, data_1], axis=-1)


def generate_samples_for_cnf(
    key: PRNGKeyArray,
    conditions: Array,
    n_samples: int,
    posrec_model: eqx.Module,
    tpc_r: float,
    radius_buffer: float,
) -> Array:
    """Generate samples from position reconstruction flow for CNF training.

    This function provides a clean interface for CNF training to sample
    from the position reconstruction flow and get properly transformed
    physical coordinates.

    Args:
        key: Random key for sampling
        conditions: Conditioning information (hit patterns)
        n_samples: Number of samples to generate
        posrec_model: Pretrained position reconstruction flow model
        tpc_r: TPC radius in cm
        radius_buffer: Buffer for predictions beyond TPC radius

    Returns:
        Array of shape (n_samples, 2) in physical coordinates
    """
    # Sample from the position reconstruction flow
    output = posrec_model.sample(key, (n_samples,), condition=conditions)

    # Transform back to physical coordinates
    return data_inv_transformation(
        jnp.reshape(output, (-1, 2)), tpc_r, radius_buffer
    )


def posrec_flow(pretrained_posrec_flow_path, config: "Config"):
    """
    Load a pretrained position reconstruction flow model, which is a coupling
    flow model with rational quadratic spline bijections. The model uses a
    standard normal base distribution.

    Parameters
    ----------
    pretrained_posrec_flow_path : str or Path
        Path to the pretrained model weights file. Should be compatible with
        equinox's tree serialization format.
    config : Config
        Configuration object containing position reconstruction flow
        parameters.

    Returns
    -------
    eqx.Module
        A pretrained coupling flow model with loaded weights.
    """
    bijection = RationalQuadraticSpline(
        knots=config.posrec.spline_knots,
        interval=config.posrec.spline_interval,
    )

    key = jax.random.PRNGKey(42)
    key, flow_key = jax.random.split(key, 2)

    posrec_model = coupling_flow(
        flow_key,
        base_dist=StandardNormal(
            2,
        ),
        invert=config.posrec.invert_bool,
        flow_layers=config.posrec.flow_layers,
        nn_width=config.posrec.nn_width,
        nn_depth=config.posrec.nn_depth,
        nn_activation=jax.nn.leaky_relu,
        cond_dim=config.posrec.cond_dim,
        transformer=bijection,
    )

    return eqx.tree_deserialise_leaves(
        pretrained_posrec_flow_path, posrec_model
    )
