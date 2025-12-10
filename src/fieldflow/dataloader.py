"""Data loading utilities for FieldFlow.

This module provides functions for loading detector data required for
CNF training, including hit patterns and charge-insensitive-volume (CIV) survival
probability maps.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from fieldflow.config import Config


def load_civ_map(file_path: str | Path) -> RegularGridInterpolator:
    """Load charge-insensitive-volume (CIV) survival probability map.

    The CIV map provides the probability that charge generated at a given
    (r, z) position survives to be detected. This is used in the likelihood
    loss to account for position-dependent detection efficiency.

    Args:
        file_path: Path to the .npz file containing CIV map data with keys
            'R' (radial coordinates), 'Z' (z coordinates), and 'vals'
            (survival probabilities).

    Returns:
        RegularGridInterpolator that evaluates CIV probability at (r, z).

    Raises:
        FileNotFoundError: If the CIV map file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CIV map file not found: {file_path}")

    with np.load(file_path) as file:
        r = file["R"]
        z = file["Z"]
        vals = file["vals"]

    civ_map = RegularGridInterpolator(
        (r, z),
        vals,
        bounds_error=False,
        fill_value=0,
    )

    return civ_map


def load_hitpatterns(
    file_path: str | Path,
    tpc_height: float,
    z_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess hit pattern data for CNF training.

    Loads detector hit patterns and their associated z coordinates, filters
    to valid TPC volume, and applies z-coordinate scaling.

    Args:
        file_path: Path to .npz file with 'z_corr' (corrected z coordinates)
            and 'condition' (hit pattern arrays).
        tpc_height: Height of TPC drift region in cm. Events with
            z < -tpc_height or z > 0 are filtered out.
        z_scale: Scaling factor for z coordinates. The scaled z is computed
            as -z / z_scale.

    Returns:
        Tuple of (z_sel, z_sel_scaled, cond_sel) where:
            - z_sel: Filtered z coordinates in cm
            - z_sel_scaled: Scaled z coordinates (used as ODE integration time)
            - cond_sel: Corresponding hit pattern conditioning vectors

    Raises:
        FileNotFoundError: If the hitpattern file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Hitpattern file not found: {file_path}")

    # Load data - direct from notebook
    data_obj = np.load(file_path)
    z = data_obj["z_corr"]
    conditions = data_obj["condition"]

    # Filter and process data - direct from notebook
    data_bool = (z > -tpc_height) & (z < 0)
    z_sel = z[data_bool]
    z_sel_scaled = -z_sel / z_scale
    cond_sel = conditions[data_bool]

    return z_sel, z_sel_scaled, cond_sel


def load_data_from_config(
    config: "Config",
    hitpattern_path: str | Path,
    civ_map_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RegularGridInterpolator]:
    """Load all training data using configuration parameters.

    Convenience function that loads both hit patterns and CIV map using
    parameters from the configuration object.

    Args:
        config: Configuration object with training.z_scale and
            experiment.tpc_height parameters.
        hitpattern_path: Path to hit pattern .npz file.
        civ_map_path: Path to CIV map .npz file.

    Returns:
        Tuple of (z_sel, z_sel_scaled, cond_sel, civ_map) containing
        the loaded and preprocessed data.
    """
    # Extract parameters from config
    z_scale = config.training.z_scale
    tpc_height = config.experiment.tpc_height

    # Load hitpattern data
    z_sel, z_sel_scaled, cond_sel = load_hitpatterns(
        hitpattern_path,
        tpc_height=tpc_height,
        z_scale=z_scale,
    )

    # Load CIV map
    civ_map = load_civ_map(civ_map_path)

    return z_sel, z_sel_scaled, cond_sel, civ_map


def create_vectorized_civ_map(civ_map: RegularGridInterpolator):
    """Create a vectorized (batched) version of the CIV map interpolator.

    Args:
        civ_map: RegularGridInterpolator for single-point CIV evaluation.

    Returns:
        Vectorized function that can evaluate CIV at multiple points
        simultaneously.
    """
    return jax.vmap(civ_map)
