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
    """Configuration for continuous normalizing flow model architecture and
    behavior.

    This class encapsulates the parameters used to define a Continuous
    Normalizing Flow model, including neural network architecture, ODE
    solver settings, and model-specific hyperparameters.

    Attributes:
        data_size: Dimensionality of the input data.
        exact_logp: Whether to use exact log probability calculation.
        width_size: Width of neural network layers.
        depth: Depth of neural network.
        scalar: Whether to use scalar field instead of vector field.
        use_pid_controller: Whether to use PIDController instead of
            ConstantStepSize.
        rtol: Relative tolerance for PIDController.
        atol: Absolute tolerance for PIDController.
        dtmax: Maximum step size for PIDController.
        dtmin: Minimum step size for PIDController.
        t0: Starting time for ODE.
        extract_t1: End time for extract phase.
        dt0: Initial time step.
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
    """Configuration for Position Reconstruction Flow model."""

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

    This class encapsulates the parameters used during the training process,
    including optimization settings, data handling, and training strategies.

    Attributes:
        seed: Random seed for training reproducibility.
        learning_rate: Initial learning rate for the optimizer.
        weight_decay: L2 regularization parameter.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        enable_scheduler: Whether to enable learning rate scheduling.
        n_samples: Number of samples per instance for likelihood estimation.
        n_train: Size of training set.
        n_test: Size of test/validation set.
        use_best: Whether to use the best model based on validation.
        curl_loss_multiplier: Coefficient for curl loss component.
        z_scale: Scaling factor for z dimension.
        multisteps_every_k: Steps for MultiSteps optimizer.
        num_devices: Number of devices to use for data parallelization.
        save_iter: Every n number of epochs to save the model
        save_file_name: Path and file name start (/path/to/model_name)
    """

    # Training process parameters
    seed: int = 42
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 2048
    enable_scheduler: bool = True

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
    """Configuration for experimental parameters.

    This class encapsulates parameters that describe the physical
    experimental setup and constraints.

    Attributes:
        tpc_height: Height of the TPC for filtering z coordinates.
        tpc_r: Radius of the TPC for boundary constraints.
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
