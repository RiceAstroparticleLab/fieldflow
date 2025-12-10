"""Utility functions for FieldFlow.

This module provides helper functions for coordinate computations and
model manipulation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


@jax.jit
def compute_r(xy_arr):
    """Compute radial distances from (x, y) coordinates.

    Args:
        xy_arr: Array of shape (N, 2) containing x, y coordinates.

    Returns:
        Array of shape (N,) containing radial distances sqrt(x² + y²).
    """
    return jnp.sqrt(xy_arr[:, 0] ** 2 + xy_arr[:, 1] ** 2)


def freeze_model_gradients(model: eqx.Module) -> eqx.Module:
    """Wrap model to prevent gradient computation through all parameters.

    This function uses equinox partitioning to separate trainable parameters
    from the static model structure, applies jax.lax.stop_gradient to the
    parameters to prevent gradient flow, and recombines them into a frozen
    model.

    This is useful for pretrained models that should not be updated during
    training of a larger system.

    Args:
        model: Equinox module to freeze

    Returns:
        Frozen model with gradient stopping applied to all parameters
    """
    # Separate trainable parameters from static structure
    params, static = eqx.partition(model, eqx.is_array)

    # Apply stop_gradient to prevent gradient computation
    frozen_params = jax.lax.stop_gradient(params)

    # Recombine into complete model
    return eqx.combine(frozen_params, static)
