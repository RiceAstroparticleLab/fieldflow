"""JAX-based continuous normalizing flows for electric field modeling.

FieldFlow provides tools for modeling electric fields in dual-phase Time
Projection Chambers (TPCs) using continuous normalizing flows (CNFs). The
architecture mirrors the physical structure of dual-phase TPCs, with separate
neural networks for the extraction field (z-independent distortions) and
drift field (z-dependent distortions).

The library supports two approaches for enforcing Maxwell's equations:

- **Scalar potential method**: Models the field as the negative gradient of a
  learned scalar potential, which is curl-free by construction.
- **Vector field with curl loss**: Directly learns the vector field while
  penalizing non-zero curl during training.
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
