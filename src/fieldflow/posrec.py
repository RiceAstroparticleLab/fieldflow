"""Pretrained conditional normalizing flow for position reconstruction from
detector hit patterns
"""

import equinox as eqx
import jax
from config import PosRecFlowConfig
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import StandardNormal
from flowjax.flows import coupling_flow


def posrec_flow(pretrained_posrec_flow_path):
    """
    Load a pretrained position reconstruction flow model, which is a coupling
    flow model with rational quadratic spline bijections. The model uses a
    standard normal base distribution.

    Parameters
    ----------
    pretrained_posrec_flow_path : str or Path
        Path to the pretrained model weights file. Should be compatible with
        equinox's tree serialization format.

    Returns
    -------
    eqx.Module
        A pretrained coupling flow model with loaded weights.
    """
    bijection = RationalQuadraticSpline(
                knots=PosRecFlowConfig.spline_knots,
                interval=PosRecFlowConfig.spline_interval
            )

    key = jax.random.PRNGKey(42)
    key, flow_key = jax.random.split(key, 2)

    posrec_model = coupling_flow(
        flow_key,
        base_dist=StandardNormal(2,),
        invert=PosRecFlowConfig.invert_bool,
        flow_layers=PosRecFlowConfig.flow_layers,
        nn_width=PosRecFlowConfig.nn_width,
        nn_depth=PosRecFlowConfig.nn_depth,
        nn_activation=jax.nn.leaky_relu,
        cond_dim=PosRecFlowConfig.cond_dim,
        transformer=bijection
        )

    return eqx.tree_deserialise_leaves(
        pretrained_posrec_flow_path, posrec_model
        )
