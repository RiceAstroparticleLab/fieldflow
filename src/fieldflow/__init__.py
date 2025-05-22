"""JAX-based normalizing flows for physical field modeling.

This package provides tools for modeling physical fields using continuous
normalizing flows, with a focus on electric field modeling for particle
detectors.
"""

from fieldflow.config import Config, load_config
from fieldflow.dataloader import load_civ_map, load_data_from_config
from fieldflow.model import ContinuousNormalizingFlow, MLPFunc
from fieldflow.posrec import generate_samples_for_cnf, posrec_flow
from fieldflow.train import (
    create_optimizer,
    likelihood_loss,
    train,
    train_model_from_config,
)

__all__ = [
    "Config",
    "ContinuousNormalizingFlow",
    "MLPFunc",
    "create_optimizer",
    "generate_samples_for_cnf",
    "likelihood_loss",
    "load_civ_map",
    "load_config",
    "load_data_from_config",
    "posrec_flow",
    "train",
    "train_model_from_config",
]
