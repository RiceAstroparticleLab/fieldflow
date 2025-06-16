"""Pretrained conditional normalizing flow for position reconstruction from
detector hit patterns, including coordinate transformations.
"""

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as special
from flowjax.bijections import (
    AbstractBijection,
    RationalQuadraticSpline,
)
from flowjax.distributions import StandardNormal
from flowjax.flows import coupling_flow
from jaxtyping import Array, PRNGKeyArray, ArrayLike
from typing import ClassVar

if TYPE_CHECKING:
    from fieldflow.config import Config

# Global constants for numerical stability - adapt to JAX precision setting
_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
TINY = jnp.finfo(_DTYPE).smallest_normal  # For preventing underflow
#EPS = 1e-7  # Avoid potential numerical issues
EPS = jnp.finfo(_DTYPE).eps  # For convergence tolerance (machine epsilon)


@jax.custom_jvp
def gammap_inverse(p: ArrayLike, a: float) -> ArrayLike:
    """
    Inverse of regularized incomplete gamma function using Halley's method.

    Finds x such that gammainc(a, x) = p.
    This is equivalent to the inverse CDF of Gamma(a, 1) distribution.
    Initial guess based on Numerical Recipes (Press et al., 2007).

    Args:
        p: Probability values in [0, 1]
        a: Shape parameter (positive)

    Returns:
        Quantiles where gammainc(a, x) = p
    """

    def objective(x):
        """F(x) = gammainc(a, x) - p"""
        return special.gammainc(a, x) - p

    # Initial guess from Numerical Recipes
    def initial_guess(u_val, a_val):
        # a = dof/2 for chi-squared

        def large_a_guess():
            # For a > 1: use Wilson-Hilferty approximation
            pp = jnp.where(u_val < 0.5, u_val, 1.0 - u_val)
            t = jnp.sqrt(-2.0 * jnp.log(pp))
            x = (2.30753 + t * 0.27061) / (
                1.0 + t * (0.99229 + t * 0.04481)
            ) - t
            x = jnp.where(u_val < 0.5, -x, x)
            return jnp.fmax(
                1e-3,
                a_val
                * (1.0 - 1.0 / (9.0 * a_val) - x / (3.0 * jnp.sqrt(a_val)))
                ** 3,
            )

        def small_a_guess():
            # For a <= 1: use equations (6.2.8) and (6.2.9)
            t = 1.0 - a_val * (0.253 + a_val * 0.12)
            return jnp.where(
                u_val < t,
                (u_val / t) ** (1.0 / a_val),
                1.0 - jnp.log(1.0 - (u_val - t) / (1.0 - t)),
            )

        return jnp.real(jnp.where(
            a_val > 1.0, large_a_guess(), small_a_guess()
            ))

    # Derivatives for Halley's method
    f = objective
    df_dx = jax.grad(objective)
    d2f_dx2 = jax.grad(df_dx)

    x = initial_guess(p, a)

    # Use while_loop for dynamic convergence
    def cond_fn(state):
        x, step, iteration = state
        # Continue while step is large and we haven't exceeded max iterations
        return (jnp.abs(step) > EPS * jnp.abs(x)) & (iteration < 12)

    def body_fn(state):
        x, _, iteration = state

        f_val = f(x)
        df_val = df_dx(x)
        d2f_val = d2f_dx2(x)

        # Halley's method: x_{n+1} = x_n - 2*f*f' / (2*f'^2 - f*f'')
        numerator = 2 * f_val * df_val
        denominator = 2 * df_val**2 - f_val * d2f_val

        # Avoid division by zero and ensure step is reasonable
        denominator = jnp.where(
            jnp.abs(denominator) < TINY,
            jnp.sign(denominator) * TINY,
            denominator,
        )

        step = numerator / denominator
        x_new = x - step

        # Ensure x stays positive
        x_new = jnp.fmax(x_new, TINY)

        return (x_new, step, iteration + 1)

    # Initial state: (x, step, iteration)
    initial_state = (x, jnp.inf, 0)
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    x = final_state[0]

    return x


@gammap_inverse.defjvp
def gammap_inverse_jvp(primals, tangents):
    """
    Custom JVP for gammap_inverse using implicit function theorem.

    For F(x, p) = gammainc(a, x) - p = 0:
    dx/dp = -∂F/∂p / ∂F/∂x = 1 / (∂/∂x gammainc(a, x))
    """
    p, a = primals
    p_dot, _ = tangents

    # Forward pass
    x = gammap_inverse(p, a)

    # Compute derivative: dx/dp = 1 / (d/dx gammainc(a, x))
    def gammainc_x(x_val):
        return special.gammainc(a, x_val)

    dgammainc_dx = jax.grad(gammainc_x)(x)

    # Avoid division by zero
    dgammainc_dx = jnp.where(
        jnp.abs(dgammainc_dx) < TINY,
        jnp.sign(dgammainc_dx) * TINY,
        dgammainc_dx,
    )

    dx_dp = 1.0 / dgammainc_dx

    # For now, ignore a derivatives (could be added if needed)
    x_dot = dx_dp * p_dot

    return x, x_dot


class StandardNormalToUnitBall(AbstractBijection):
    """
    Bijection that maps standard normal distribution to uniform on unit ball.

    This bijection leverages the fact that for a d-dimensional standard normal
    vector x:
    - ||x||² ~ χ²(d) (chi-squared with d degrees of freedom)
    - The direction x/||x|| is uniform on the unit sphere

    The transformation:
    1. Computes squared magnitude r² = ||x||²
    2. Applies chi-squared CDF: u = CDF_χ²(r²) → uniform [0,1]
    3. Transforms to correct radial distribution: r_new = u^(1/d)
    4. Scales the vector: y = (r_new / ||x||) · x

    This maps N(0,I) → Uniform(unit ball).
    """

    shape: tuple[int, ...] = (2,)  # Default to 2D
    cond_shape: ClassVar[None] = None

    def transform(self, x: ArrayLike, condition=None) -> ArrayLike:  # noqa: ARG002
        """Transform standard normal to uniform on unit ball.

        Args:
            x: Input vector from standard normal distribution
            condition: Unused (for compatibility)

        Returns:
            Vector uniformly distributed on unit ball
        """
        # Compute squared magnitude and radius
        r_squared = jnp.sum(x**2)
        r = jnp.sqrt(jnp.fmax(r_squared, TINY))

        # Apply chi-squared CDF to get uniform [0,1]
        u = special.gammainc(self.shape[0] / 2, r_squared / 2)
        u = jnp.fmax(u, TINY)  # Only prevent u = 0

        # Transform to correct radial distribution for unit ball
        # For uniform on d-dimensional ball, we want r_new = u^(1/d)
        r_new = u ** (1.0 / self.shape[0])

        # Scale the vector: preserve direction, change magnitude
        scaling_factor = r_new / r
        y = scaling_factor * x

        return y

    def inverse(self, y: ArrayLike, condition=None) -> ArrayLike:  # noqa: ARG002
        """Inverse transform: uniform on unit ball to standard normal.

        Args:
            y: Vector uniformly distributed on unit ball
            condition: Unused

        Returns:
            Vector from standard normal distribution
        """
        # Compute current radius
        r_new = jnp.sqrt(jnp.fmax(jnp.sum(y**2), TINY))

        # Inverse of radial transformation: r_new = u^(1/d) → u = r_new^d
        u = jnp.fmax(r_new ** self.shape[0], TINY)

        # Inverse chi-squared CDF to get original squared magnitude
        r_squared = 2 * gammap_inverse(u, self.shape[0] / 2)
        r = jnp.sqrt(jnp.fmax(r_squared, TINY))

        # Scale back: preserve direction, change magnitude
        scaling_factor = r / r_new
        x = scaling_factor * y

        return x

    def transform_and_log_det(
        self,
        x: ArrayLike,
        condition=None,  # noqa: ARG002
    ) -> tuple[ArrayLike, ArrayLike]:
        """Transform and compute log determinant simultaneously."""
        # Forward transformation
        r_squared = jnp.sum(x**2)
        r = jnp.sqrt(jnp.fmax(r_squared, TINY))

        u = special.gammainc(self.shape[0] / 2, r_squared / 2)
        u = jnp.fmax(u, TINY)
        r_new = u ** (1.0 / self.shape[0])

        y = (r_new / r) * x

        # Log determinant computation using JAX autodiff
        # Use jacfwd on the existing transform method
        jacobian = jax.jacfwd(self.transform)(x)
        log_det = jnp.log(jnp.abs(jnp.linalg.det(jacobian)))

        return y, log_det

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        condition=None,  # noqa: ARG002
    ) -> tuple[ArrayLike, ArrayLike]:
        """Inverse transform and compute log determinant."""
        # Forward pass of inverse
        r_new = jnp.sqrt(jnp.fmax(jnp.sum(y**2), TINY))
        u = jnp.fmax(r_new ** self.shape[0], TINY)
        r_squared = 2 * gammap_inverse(u, self.shape[0] / 2)
        r = jnp.sqrt(jnp.fmax(r_squared, TINY))

        x = (r / r_new) * y

        # Log determinant computation using JAX autodiff
        # Use jacfwd on the existing inverse method
        jacobian = jax.jacfwd(self.inverse)(y)
        log_det = jnp.log(jnp.abs(jnp.linalg.det(jacobian)))

        return x, log_det


constrain_vec = jax.vmap(StandardNormalToUnitBall.inverse)

@jax.jit
def data_inv_transformation(
    data: Array, tpc_r: float, radius_buffer: float
) -> Array:
    """Transform flow coordinates back to physical (x,y) coordinates.

    This function applies the full coordinate transformation chain to convert
    from normalized flow space back to physical detector coordinates.

    Args:
        data: Array of shape (N, 2) in flow coordinate space
        tpc_r: TPC radius in cm
        radius_buffer: Buffer for predictions beyond TPC radius

    Returns:
        Array of shape (N, 2) in physical coordinates (cm)
    """

    # Apply the constraint transformation (StandardNormalToUnitBall)
    constrained_data = constrain_vec(data)

    # Convert from [0,1] space to physical coordinates
    max_pred = tpc_r + radius_buffer
    data_0 = constrained_data[:, 0] * max_pred
    data_1 = constrained_data[:, 1] * max_pred
    return jnp.stack([data_0, data_1], axis=-1)


def generate_samples_for_cnf(
    key: PRNGKeyArray,
    conditions: Array,
    n_samples: int,
    posrec_model: eqx.Module,
    tpc_r: float = 129.96,  # Default matches experiment.tpc_r
    radius_buffer: float = 0.0,  # Default matches posrec.radius_buffer
) -> Array:
    """Generate samples from position reconstruction flow for CNF training.

    This function provides a clean interface for CNF training to sample
    from the position reconstruction flow and get properly transformed
    physical coordinates.

    Args:
        key: Random key for sampling
        conditions: Conditioning information (hit patterns)
        n_samples: Number of samples to generate
        posrec_model: Pretrained position reconstruction flow model
        tpc_r: TPC radius in cm (default: 66.4)
        radius_buffer: Buffer for predictions beyond TPC radius (default: 20.0)

    Returns:
        Array of shape (n_samples, 2) in physical coordinates
    """
    # Sample from the position reconstruction flow
    output = posrec_model.sample(key, (n_samples,), condition=conditions)

    # Transform back to physical coordinates
    return data_inv_transformation(
        jnp.reshape(output, (-1, 2)), tpc_r, radius_buffer
    )


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
        interval=config.posrec.spline_interval,
    )

    key = jax.random.key(42)
    key, flow_key = jax.random.split(key, 2)

    posrec_model = coupling_flow(
        flow_key,
        base_dist=StandardNormal((2,)),
        invert=config.posrec.invert_bool,
        flow_layers=config.posrec.flow_layers,
        nn_width=config.posrec.nn_width,
        nn_depth=config.posrec.nn_depth,
        nn_activation=jax.nn.leaky_relu,
        cond_dim=config.posrec.cond_dim,
        transformer=bijection,
    )

    return eqx.tree_deserialise_leaves(
        pretrained_posrec_flow_path, posrec_model
    )
