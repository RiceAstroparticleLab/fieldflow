"""Entry point for FieldFlow training.

This module provides a simple command-line interface for training FieldFlow
models from configuration, with optional fine-tuning of pre-trained models.
"""

import argparse
from pathlib import Path

import diffrax
import equinox as eqx
import jax

from fieldflow.config import load_config
from fieldflow.dataloader import load_data_from_config
from fieldflow.model import (
    ContinuousNormalizingFlow,
    DriftFromPotential,
    MLPFunc,
)
from fieldflow.posrec import posrec_flow
from fieldflow.train import save_model, train_model_from_config


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
            dtmin=config.model.dtmin,
        )
    else:
        step_size_controller = diffrax.ConstantStepSize()

    # Choose drift function based on whether scalar field is used
    drift_func = DriftFromPotential if config.model.scalar else MLPFunc

    # Create and return model
    return ContinuousNormalizingFlow(
        func=drift_func,
        data_size=config.model.data_size,
        exact_logp=config.model.exact_logp,
        width_size=config.model.width_size,
        depth=config.model.depth,
        key=key,
        stepsizecontroller=step_size_controller,
        t0=config.model.t0,
        dt0=config.model.dt0,
        extract_t1 = config.model.extract_t1,
    )


def main():
    """Main entry point for training FieldFlow models."""
    parser = argparse.ArgumentParser(description="FieldFlow training")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument(
        "--pretrained",
        help="Path to pretrained model for fine-tuning (optional)",
    )
    parser.add_argument("--output", help="Output directory")
    parser.add_argument(
        "--hitpatterns", help="Path to hitpatterns data file (.npz)"
    )
    parser.add_argument("--civ-map", help="Path to CIV map file (.npz)")
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
    output_path = Path(args.output) if args.output else Path(".")

    # Set up data paths
    hitpatterns_path = (
        Path(args.hitpatterns)
        if args.hitpatterns
        else Path("data/hitpatterns.npz")
    )
    civ_map_path = (
        Path(args.civ_map) if args.civ_map else Path("data/civ_map.npz")
    )
    posrec_model_path = (
        Path(args.posrec_model)
        if args.posrec_model
        else Path("data/posrec_model.eqx")
    )

    # Check if data files exist
    if not hitpatterns_path.exists():
        raise FileNotFoundError(
            f"Hitpatterns file not found: {hitpatterns_path}"
        )
    if not civ_map_path.exists():
        raise FileNotFoundError(f"CIV map file not found: {civ_map_path}")
    if not posrec_model_path.exists():
        raise FileNotFoundError(
            f"Position reconstruction model not found: {posrec_model_path}"
        )

    # Initialize random key
    key = jax.random.key(config.training.seed)
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
        model = eqx.tree_deserialise_leaves(pretrained_path, dummy_model)
    else:
        print("Creating new model from scratch")
        model = create_model_from_config(config, subkey)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Train model
    print("Starting training...")
    trained_model, train_losses, test_losses, best_epoch = (
        train_model_from_config(
            key=key,
            model=model,
            conditions=cond_sel,
            t1s=z_sel_scaled,
            zs=z_sel,
            posrec_model=posrec_model,
            civ_map=civ_map,
            config=config,
            output_path=str(output_path),
        )
    )

    # Save model, train, and test losses
    save_file_name = config.training.save_file_name
    save_model(
        trained_model,
        str(output_path / f"best_{save_file_name}_epoch_{best_epoch}.eqx"),
    )
    jax.numpy.savez(
        str(output_path / "train_losses.npz"), train_losses=train_losses
    )
    jax.numpy.savez(
        str(output_path / "val_losses.npz"), test_losses=test_losses
    )
    print(f"Training complete. Final validation loss: {test_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
