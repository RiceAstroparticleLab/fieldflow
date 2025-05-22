"""Data loading utilities for FieldFlow.

This module provides functions for loading hitpattern data and CIV maps,
extracted directly from the working notebook code.
"""

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator


if TYPE_CHECKING:
    from fieldflow.config import Config


def load_civ_map(file_path: str | Path) -> RegularGridInterpolator:
    """Load CIV map from gzipped JSON file.

    Direct adaptation of load_civ() function from notebook.

    Args:
        file_path: Path to the gzipped JSON file containing CIV map data

    Returns:
        RegularGridInterpolator function for CIV map
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CIV map file not found: {file_path}")

    with gzip.open(file_path, "rb") as f:
        file = json.load(f)

    civ_map = RegularGridInterpolator(
        tuple([np.linspace(*ax[1]) for ax in file["coordinate_system"]]),
        np.array(file["survival_probability_map"]).reshape(
            [ax[1][-1] for ax in file["coordinate_system"]]
        ),
        bounds_error=False,
        fill_value=0,
    )

    return civ_map


def load_hitpatterns(
    file_path: str | Path,
    tpc_height: float,
    z_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and process hitpattern data.

    Direct extraction from notebook data loading and processing code.

    Args:
        file_path: Path to the .npz file containing hitpattern data
        tpc_height: Height of the TPC for filtering z coordinates
        z_scale: Scaling factor for z coordinates

    Returns:
        Tuple of (z_sel, z_sel_scaled, cond_sel)
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
    """Load all data using configuration parameters.

    Args:
        config: Configuration object containing training parameters
        hitpattern_path: Path to hitpattern .npz file
        civ_map_path: Path to CIV map JSON file

    Returns:
        Tuple of (z_sel, z_sel_scaled, cond_sel, civ_map)
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
    """Create vectorized version of CIV map interpolator.

    Direct from notebook: vec_civ_map = jax.vmap(civ_map)

    Args:
        civ_map: RegularGridInterpolator for CIV map

    Returns:
        Vectorized CIV map function
    """
    return jax.vmap(civ_map)
