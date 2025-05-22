"""Pretrained conditional normalizing flow for position reconstruction from
detector hit patterns
"""

from typing import TYPE_CHECKING

import equinox as eqx
import jax
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import StandardNormal
from flowjax.flows import coupling_flow

if TYPE_CHECKING:
    from config import Config


def posrec_flow(pretrained_posrec_flow_path, config: "Config"):
    """
    Load a pretrained position reconstruction flow model, which is a coupling
    flow model with rational quadratic spline bijections. The model uses a
    standard normal base distribution.

    Parameters
    ----------
    pretrained_posrec_flow_path : str or Path
        Path to the pretrained model weights file. Should be compatible with
        equinox's tree serialization format.
    config : Config
        Configuration object containing position reconstruction flow
        parameters.

    Returns
    -------
    eqx.Module
        A pretrained coupling flow model with loaded weights.
    """
    bijection = RationalQuadraticSpline(
                knots=config.posrec.spline_knots,
                interval=config.posrec.spline_interval
            )

    key = jax.random.PRNGKey(42)
    key, flow_key = jax.random.split(key, 2)

    posrec_model = coupling_flow(
        flow_key,
        base_dist=StandardNormal(2,),
        invert=config.posrec.invert_bool,
        flow_layers=config.posrec.flow_layers,
        nn_width=config.posrec.nn_width,
        nn_depth=config.posrec.nn_depth,
        nn_activation=jax.nn.leaky_relu,
        cond_dim=config.posrec.cond_dim,
        transformer=bijection
        )

    return eqx.tree_deserialise_leaves(
        pretrained_posrec_flow_path, posrec_model
        )
