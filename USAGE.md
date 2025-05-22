# FieldFlow Usage Guide

## Training a New Model

To train a new model from scratch, use the following command:

```bash
python -m fieldflow config.toml
```

This will:
1. Load the configuration from `config.toml`
2. Create a new model from random initialization
3. Train the model using the specified parameters
4. Save the trained model to `model.eqx` (default)

## Fine-Tuning a Pre-trained Model

To fine-tune an existing model, use the `--pretrained` parameter:

```bash
python -m fieldflow config.toml --pretrained pretrained_model.eqx
```

This will:
1. Load the configuration from `config.toml`
2. Load the pretrained model from `pretrained_model.eqx`
3. Continue training from the loaded checkpoint
4. Save the fine-tuned model to `model.eqx` (default)

## Additional Parameters

```
python -m fieldflow --help
```

```
usage: __main__.py [-h] [--pretrained PRETRAINED] [--output OUTPUT]
                   [--hitpatterns HITPATTERNS] [--civ-map CIV_MAP]
                   [--posrec-model POSREC_MODEL]
                   config

FieldFlow training

positional arguments:
  config                Path to configuration file

options:
  -h, --help            show this help message and exit
  --pretrained PRETRAINED
                        Path to pretrained model for fine-tuning (optional)
  --output OUTPUT       Output path for trained model (default: model.eqx)
  --hitpatterns HITPATTERNS
                        Path to hitpatterns data file (.npz)
  --civ-map CIV_MAP     Path to CIV map file (.json.gz)
  --posrec-model POSREC_MODEL
                        Path to pretrained position reconstruction model (.eqx)
```

## Default File Paths

If not specified, the following default paths are used:
- Hitpatterns: `data/hitpatterns.npz`
- CIV map: `data/civ_map.json.gz`
- Position reconstruction model: `data/posrec_model.eqx`
- Output model: `model.eqx`

## Example Configuration (TOML)

```toml
[model]
data_size = 2
exact_logp = true
width_size = 48
depth = 3
use_pid_controller = true
rtol = 1e-3
atol = 1e-6
dtmax = 5.0
t0 = 0.0
extract_t1 = 10.0
dt0 = 1.0

[training]
seed = 42
learning_rate = 2e-3
weight_decay = 1e-4
epochs = 100
batch_size = 2048
n_samples = 16
n_train = 200000
n_test = 20000
use_best = true
curl_loss_multiplier = 1000.0
z_scale = 5.0
multisteps_every_k = 4

[experiment]
tpc_height = 148.6515
tpc_r = 66.4

[posrec]
flow_layers = 5
nn_width = 128
nn_depth = 3
invert_bool = false
cond_dim = 494
spline_knots = 5
spline_interval = 5.0
radius_buffer = 20.0
```
