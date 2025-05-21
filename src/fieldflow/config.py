"""Configuration classes for FieldFlow models and training.

This module provides configuration dataclasses that encapsulate the parameters
for model architecture and training workflows for normalizing flow models.
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ModelConfig:
    """Configuration for CNF model architecture and behavior.

    This class encapsulates the parameters used to define a Continuous
    Normalizing Flow (CNF) model, including neural network architecture, ODE
    solver settings, and model-specific hyperparameters.

    Attributes:
        data_size: Dimensionality of the input data.
        exact_logp: Whether to use exact log probability calculation.
        width_size: Width of neural network layers.
        depth: Depth of neural network.
        func_class: The function class to use ("MLPFunc" or "Func").
        use_pid_controller: Whether to use PIDController instead of
            ConstantStepSize.
        rtol: Relative tolerance for PIDController.
        atol: Absolute tolerance for PIDController.
        dtmax: Maximum step size for PIDController.
        t0: Starting time for ODE.
        extract_t1: End time for extract phase.
        dt0: Initial time step.
    """

    data_size: int = 2
    exact_logp: bool = True
    width_size: int = 48
    depth: int = 3
    func_class: Literal["MLPFunc", "Func"] = "MLPFunc"

    # ODE Solver settings
    use_pid_controller: bool = True
    rtol: float = 1e-3
    atol: float = 1e-6
    dtmax: float = 5.0

    # Time settings
    t0: float = 0.0
    extract_t1: float = 10.0
    dt0: float = 1.0

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ModelConfig":
        """Create a ModelConfig instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            A ModelConfig instance with the specified parameters.
        """
        return cls(**config_dict)


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
        n_samples: Number of samples per instance for likelihood estimation.
        n_train: Size of training set.
        n_test: Size of test/validation set.
        use_best: Whether to use the best model based on validation.
        curl_loss_multiplier: Coefficient for curl loss component.
        z_scale: Scaling factor for z dimension.
    """

    # Training process parameters
    seed: int = 42
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 2048

    # Data and sampling parameters
    n_samples: int = 16
    n_train: int = 200000
    n_test: int = 20000

    # Model selection and loss parameters
    use_best: bool = True
    curl_loss_multiplier: float = 1000.0
    z_scale: float = 5.0

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingConfig":
        """Create a TrainingConfig instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            A TrainingConfig instance with the specified parameters.
        """
        return cls(**config_dict)
