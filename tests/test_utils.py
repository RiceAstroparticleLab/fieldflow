"""Tests for utility functions"""

import jax.numpy as jnp
import pytest

from fieldflow.utils import compute_r


def test_compute_r():
    """Test compute_r with various coordinate pairs."""
    xy = jnp.array(
        [
            [3.0, 4.0],  # radius 5
            [0.0, 0.0],  # radius 0 (origin)
            [1.0, 0.0],  # radius 1 (on axis)
            [-3.0, -4.0],  # radius 5 (negative coords)
            [0.0, 2.0],  # radius 2 (single axis)
        ]
    )
    expected = [5.0, 0.0, 1.0, 5.0, 2.0]
    result = compute_r(xy)

    assert result == pytest.approx(expected)
