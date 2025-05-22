[![CI](https://github.com/RiceAstroparticleLab/fieldflow/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/RiceAstroparticleLab/fieldflow/actions/workflows/lint.yml)

# FieldFlow

JAX-based continuous normalizing flows for electric field modeling in Time Projection Chambers (TPCs). FieldFlow uses neural ODEs to learn the mapping between hit patterns and electric field configurations for astrophysical particle detection experiments.

The library implements continuous normalizing flows with configurable ODE solvers, supporting both exact and approximate log probability computation. Multi-GPU training is supported with automatic data parallelization across devices.

## Installation

Requires Python ≥3.10. Install from source:

```bash
git clone https://github.com/RiceAstroparticleLab/fieldflow.git
cd fieldflow
pip install -e .
```

## Quick Usage

Train a new model:
```bash
python -m fieldflow config.toml
```

Fine-tune a pretrained model:
```bash
python -m fieldflow config.toml --pretrained model.eqx
```

See [USAGE.md](USAGE.md) for detailed usage instructions and [sample_config.toml](sample_config.toml) for configuration options.

## Key Features

- **Continuous normalizing flows** with exact or approximate log probability computation
- **Multi-GPU training** with automatic data sharding across devices
- **Configurable ODE solvers** including adaptive PID controllers
- **Fine-tuning support** for transfer learning from pretrained models
- **Position reconstruction** integration with pretrained flow models
- **JAX-based** implementation for GPU acceleration and automatic differentiation

## License

MIT License. See [LICENSE](LICENSE) for details.

## Issues & Contact

Report issues at [GitHub Issues](https://github.com/RiceAstroparticleLab/fieldflow/issues).
