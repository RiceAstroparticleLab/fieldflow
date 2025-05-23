"""Entry point for FieldFlow training.

This module provides a simple command-line interface for training FieldFlow
models from configuration, with optional fine-tuning of pre-trained models.
"""

import argparse
import os
from pathlib import Path

import diffrax
import equinox as eqx
import jax

from fieldflow.config import load_config
from fieldflow.dataloader import load_data_from_config
from fieldflow.model import ContinuousNormalizingFlow
from fieldflow.posrec import posrec_flow
from fieldflow.train import train_model_from_config


def create_model_from_config(config, key):
    """Create a CNF model from configuration.

    Args:
        config: Configuration object containing model parameters
        key: JAX PRNG key for model initialization

    Returns:
        Initialized ContinuousNormalizingFlow model
    """
    # Create step size controller based on config
    if config.model.use_pid_controller:
        step_size_controller = diffrax.PIDController(
            rtol=config.model.rtol,
            atol=config.model.atol,
            dtmax=config.model.dtmax,
        )
    else:
        step_size_controller = diffrax.ConstantStepSize()

    # Create and return model
    return ContinuousNormalizingFlow(
        data_size=config.model.data_size,
        exact_logp=config.model.exact_logp,
        width_size=config.model.width_size,
        depth=config.model.depth,
        key=key,
        stepsizecontroller=step_size_controller,
        t0=config.model.t0,
        dt0=config.model.dt0,
    )


def save_model(model, path):
    """Save model to disk.

    Args:
        model: Trained model to save
        path: Output path for the saved model
    """
    eqx.tree_serialise_leaves(path, model)
    print(f"Model saved to {path}")


def main():
    """Main entry point for training FieldFlow models."""
    parser = argparse.ArgumentParser(description="FieldFlow training")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument(
        "--pretrained",
        help="Path to pretrained model for fine-tuning (optional)",
    )
    parser.add_argument(
        "--output", help="Output path for trained model (default: model.eqx)"
    )
    parser.add_argument(
        "--hitpatterns", help="Path to hitpatterns data file (.npz)"
    )
    parser.add_argument("--civ-map", help="Path to CIV map file (.json.gz)")
    parser.add_argument(
        "--posrec-model",
        help="Path to pretrained position reconstruction model (.eqx)",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Set default output path if not provided
    output_path = args.output or "model.eqx"

    # Set up data paths
    hitpatterns_path = args.hitpatterns or "data/hitpatterns.npz"
    civ_map_path = args.civ_map or "data/civ_map.json.gz"
    posrec_model_path = args.posrec_model or "data/posrec_model.eqx"

    # Check if data files exist
    if not os.path.exists(hitpatterns_path):
        raise FileNotFoundError(
            f"Hitpatterns file not found: {hitpatterns_path}"
        )
    if not os.path.exists(civ_map_path):
        raise FileNotFoundError(f"CIV map file not found: {civ_map_path}")
    if not os.path.exists(posrec_model_path):
        raise FileNotFoundError(
            f"Position reconstruction model not found: {posrec_model_path}"
        )

    # Initialize random key
    key = jax.random.PRNGKey(config.training.seed)
    key, subkey = jax.random.split(key)

    # Load data
    print(f"Loading data from {hitpatterns_path} and {civ_map_path}")
    z_sel, z_sel_scaled, cond_sel, civ_map = load_data_from_config(
        config, hitpatterns_path, civ_map_path
    )

    # Load pretrained position reconstruction model
    print(f"Loading position reconstruction model from {posrec_model_path}")
    posrec_model = posrec_flow(posrec_model_path, config)

    # Create model or load pretrained model
    if args.pretrained:
        pretrained_path = Path(args.pretrained)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained model not found: {pretrained_path}"
            )

        print(
            f"Loading pretrained model from {pretrained_path} for fine-tuning"
        )
        # Create a dummy model with the right structure first
        dummy_model = create_model_from_config(config, subkey)
        # Then load the pretrained weights into it
        with open(pretrained_path, "rb") as f:
            model = eqx.tree_deserialise_leaves(f.read(), dummy_model)
    else:
        print("Creating new model from scratch")
        model = create_model_from_config(config, subkey)

    # Train model
    print("Starting training...")
    trained_model, train_losses, test_losses = train_model_from_config(
        key=key,
        model=model,
        conditions=cond_sel,
        t1s=z_sel_scaled,
        zs=z_sel,
        posrec_model=posrec_model,
        civ_map=civ_map,
        config=config,
    )

    # Save model
    save_model(trained_model, output_path)

    print(f"Training complete. Final test loss: {test_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
