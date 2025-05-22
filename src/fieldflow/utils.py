"""Utility functions"""

import jax
import jax.numpy as jnp


@jax.jit
def compute_r(xy_arr):
    """Compute radii from (x,y) coordinates.

    Direct extraction from notebook code.

    Args:
        xy_arr: Array of shape (N, 2) containing x,y coordinates

    Returns:
        Array of shape (N,) containing computed radii
    """
    return jnp.sqrt(xy_arr[:, 0]**2 + xy_arr[:, 1]**2)
