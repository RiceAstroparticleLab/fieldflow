"""Configuration classes for FieldFlow models and training.

This module provides configuration dataclasses that encapsulate the parameters
for model architecture and training workflows.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import the appropriate TOML parser based on Python version
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class ModelConfig:
    """Configuration for continuous normalizing flow model architecture.

    This class defines parameters for the CNF model that learns electric field
    distortions in dual-phase TPCs. The model uses separate neural networks for
    extraction (z-independent) and drift (z-dependent) field components.

    Attributes:
        data_size: Dimensionality of the input data (default 2 for x,y).
        exact_logp: If True, compute exact log probability using full Jacobian
            trace. If False, use Hutchinson trace estimator (faster but
            approximate).
        width_size: Width of hidden layers in the neural networks.
        depth: Number of hidden layers in the neural networks.
        scalar: If True, use scalar potential method (curl-free by
            construction). If False, use direct vector field with curl
            penalty loss.
        use_pid_controller: If True, use adaptive PID step size controller.
            If False, use constant step size.
        rtol: Relative tolerance for adaptive ODE solver.
        atol: Absolute tolerance for adaptive ODE solver.
        dtmax: Maximum step size for adaptive ODE solver.
        dtmin: Minimum step size for adaptive ODE solver.
        t0: Starting time for ODE integration.
        extract_t1: End time for extraction phase ODE integration.
        dt0: Initial time step for ODE solver.
    """

    data_size: int = 2
    exact_logp: bool = True
    width_size: int = 192
    depth: int = 10
    scalar: bool = False

    # ODE Solver settings
    use_pid_controller: bool = True
    rtol: float = 1e-3
    atol: float = 1e-6
    dtmax: float = 2.0
    dtmin: float = 0.05

    # Time settings
    t0: float = 0.0
    extract_t1: float = 10.0
    dt0: float = 1.0


@dataclass
class PosRecFlowConfig:
    """Configuration for the position reconstruction normalizing flow.

    The position reconstruction flow is a pretrained conditional normalizing
    flow that maps detector hit patterns to (x, y) position distributions.
    This flow provides the prior distribution for CNF training.

    Attributes:
        flow_layers: Number of coupling layers in the normalizing flow.
        nn_width: Width of hidden layers in coupling layer neural networks.
        nn_depth: Number of hidden layers in coupling layer neural networks.
        invert_bool: Whether to invert the flow direction.
        cond_dim: Dimension of the conditioning vector (hit pattern size).
        spline_knots: Number of knots for rational quadratic spline bijections.
        spline_interval: Interval parameter for spline transformations.
        radius_buffer: Buffer beyond TPC radius for coordinate transformation,
            allowing predictions slightly outside the physical boundary.
    """

    # Neural network architecture
    flow_layers: int = 5
    nn_width: int = 128
    nn_depth: int = 3
    invert_bool: bool = False

    # Conditioning
    cond_dim: int = 860  # Should be set as conditions.shape[1]

    # Spline parameters
    spline_knots: int = 5
    spline_interval: float = 5.0

    # Coordinate transformation parameters
    radius_buffer: float = 20.0  # Buffer for predictions beyond TPC radius


@dataclass
class TrainingConfig:
    """Configuration for training CNF models.

    This class defines parameters for the training process including
    optimization, batching, loss computation, and multi-GPU parallelization.

    Attributes:
        seed: Random seed for reproducibility.
        learning_rate: Initial learning rate for AdamW optimizer.
        weight_decay: L2 regularization coefficient.
        epochs: Number of training epochs.
        batch_size: Number of samples per training batch.
        enable_scheduler: If True, use learning rate schedule that reduces
            LR at epochs 20 and 70. If False, use constant learning rate.
        epoch_start: Starting epoch number (useful for resuming training).
        n_samples: Number of Monte Carlo samples per event for likelihood
            estimation from the position reconstruction flow.
        n_train: Number of training samples to use from the dataset.
        n_test: Number of samples for validation/test set.
        use_best: If True, return the model with lowest validation loss.
        curl_loss_multiplier: Weight for curl penalty term (only used when
            scalar=False in ModelConfig).
        z_scale: Scaling factor to normalize z coordinates.
        multisteps_every_k: Gradient accumulation steps before optimizer
            update.
        num_devices: Number of GPUs for data parallelization.
        save_iter: Save model checkpoint every N epochs.
        save_file_name: Base filename for saved model checkpoints.
    """

    # Training process parameters
    seed: int = 42
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 2048
    enable_scheduler: bool = True
    epoch_start: int = 0

    # Data and sampling parameters
    n_samples: int = 16
    n_train: int = 65536
    n_test: int = 4096

    # Model selection and loss parameters
    use_best: bool = True
    curl_loss_multiplier: float = 1000.0
    z_scale: float = 5.0
    multisteps_every_k: int = 1

    # Distributed training parameters
    num_devices: int = 1

    # Model saving
    save_iter: int = 2
    save_file_name: str = "model"

@dataclass
class ExperimentConfig:
    """Configuration for the physical TPC geometry.

    This class defines physical parameters of the dual-phase Time Projection
    Chamber that constrain the model.

    Attributes:
        tpc_height: Height of the TPC drift region in cm. Used to filter
            events by z coordinate (keeping -tpc_height < z < 0).
        tpc_r: Radius of the cylindrical TPC in cm. Used for boundary
            constraints in the loss function.
    """

    tpc_height: float = 259.92
    tpc_r: float = 129.96  # TPC radius in cm


@dataclass
class Config:
    """Main configuration class for FieldFlow.

    This top-level configuration class combines all configuration components,
    providing a unified interface for configuring the entire system.

    Attributes:
        model: Model architecture configuration.
        training: Training process configuration.
        experiment: Experimental setup configuration.
        posrec: Position reconstruction flow configuration.
        experiment_name: Optional name for the experiment.
        description: Optional description of the experiment.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    posrec: PosRecFlowConfig = field(default_factory=PosRecFlowConfig)
    experiment_name: str | None = None
    description: str | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary.

        This method handles nested configuration dictionaries by automatically
        converting them to the appropriate dataclass types.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            A Config instance with the specified parameters.
        """
        # Handle nested configs
        model_dict = config_dict.get("model", {})
        training_dict = config_dict.get("training", {})
        experiment_dict = config_dict.get("experiment", {})
        posrec_dict = config_dict.get("posrec", {})

        # Get remaining kwargs excluding nested configs
        remaining_kwargs = {
            k: v
            for k, v in config_dict.items()
            if k not in ("model", "training", "experiment", "posrec")
        }

        # Create the config with nested objects
        return cls(
            model=ModelConfig(**model_dict),
            training=TrainingConfig(**training_dict),
            experiment=ExperimentConfig(**experiment_dict),
            posrec=PosRecFlowConfig(**posrec_dict),
            **remaining_kwargs,
        )

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "Config":
        """Load configuration from a TOML file.

        Args:
            config_path: Path to the TOML configuration file.

        Returns:
            A Config instance with parameters loaded from the TOML file.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError("Config file not found")

        with open(config_path, "rb") as f:
            config_dict = tomllib.load(f)
        return cls.from_dict(config_dict)


def load_config(config_path: str | Path) -> Config:
    """Load configuration from a TOML file.

    Args:
        config_path: Path to configuration file

    Returns:
        A Config instance with the loaded configuration
    """
    with open(Path(config_path), "rb") as f:
        config_dict = tomllib.load(f)

    return Config.from_dict(config_dict)
