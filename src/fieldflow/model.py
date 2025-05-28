"""
Model definitions for continuous normalizing flow for electric field modeling.
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp


def approx_logp_wrapper(t, y, args):
    """Wrapper function for approximating log probability using the
      Hutchinson trace estimator.

    Args:
        t (float): Current time in the ODE integration.
        y (tuple): Tuple containing (state, log_probability) where
            state is the current data point and log_probability is
            the accumulated log determinant.
        args (tuple): Arguments containing (*model_args, eps, func)
            where eps is the random vector for Hutchinson estimation
            and func is the drift function.

    Credit: this function was adapted from the FFJORD repo.

    Returns:
        tuple: (f, logp) where f (drift fn) is the vector field and
        logp is the trace approximation.
    """
    y, _ = y
    *args, eps, func = args

    def fn(y):
        return func(t, y, args)

    f, vjp_fn = jax.vjp(fn, y)
    (eps_dfdy,) = vjp_fn(eps)
    logp = jnp.sum(eps_dfdy * eps)
    return f, logp


def exact_logp_wrapper(t, y, args):
    """Wrapper function for exact log probability computation using full
    Jacobian trace.

    Args:
        t (float): Current time in the ODE integration.
        y (tuple): Tuple containing (state, log_probability) where state is the
          current data point and log_probability is the accumulated log
          determinant.
        args (tuple): Arguments containing (*model_args, _, func) where func is
          the drift function.

    Returns:
        tuple: (drift, log_probability_derivative) where drift is the vector
          field and log_probability_derivative is the exact trace.
    """
    y, _ = y
    *args, _, func = args

    def fn(y):
        return func(t, y, args)

    f, vjp_fn = jax.vjp(fn, y)
    (size,) = y.shape  # this implementation only works for 1D input
    eye = jnp.eye(size)
    (dfdy,) = jax.vmap(vjp_fn)(eye)
    logp = jnp.trace(dfdy)
    return f, logp


class MLPFunc(eqx.Module):
    """Multilayer perceptron that models the drift function in
    a continuous normalizing flow.
    """

    layers: list[eqx.nn.Linear]

    def __init__(self, *, data_size, width_size, depth, key, **kwargs):
        """Initialize the MLP drift function.

        Args:
            data_size (int): Dimensionality of the input data.
            width_size (int): Hidden layer width. Ignored if depth=0.
            depth (int): Number of hidden layers. If 0, creates a single linear
              layer.
            key (jax.random.key): Random key for parameter initialization.
        """
        super().__init__(**kwargs)
        keys = jax.random.split(key, depth + 1)
        layers = []
        # layers_t = []
        if depth == 0:
            layers.append(eqx.nn.Linear(data_size + 1, data_size, key=keys[0]))
        else:
            layers.append(
                eqx.nn.Linear(data_size + 1, width_size, key=keys[0])
            )
            for i in range(depth - 1):
                layers.append(
                    eqx.nn.Linear(width_size, width_size, key=keys[i + 1])
                )
            layers.append(eqx.nn.Linear(width_size, data_size, key=keys[-1]))
        self.layers = layers

    def __call__(self, t, y, args):  # noqa: ARG002
        """Compute the drift (velocity) at given time and state.

        Args:
            t (float): Current time.
            y (jnp.ndarray): Current state vector.

        Returns:
            jnp.ndarray: Drift vector of the same shape as y.
        """
        t = jnp.asarray(t)[None]
        y = jnp.concatenate((y, t), axis=-1)

        for layer in self.layers[:-1]:
            y = layer(y)
            y = jax.nn.silu(y)
        y = self.layers[-1](y)
        return y


class ContinuousNormalizingFlow(eqx.Module):
    """Continuous normalizing flow using neural ODEs.

    Attributes:
        func_drift (eqx.Module): Neural network modeling the drift function.
        data_size (int): Dimensionality of the data.
        exact_logp (bool): Whether to use exact log probability computation.
        t0 (float): Initial time for ODE integration.
        dt0 (float): Initial time step for ODE integration.
        stepsizecontroller (diffrax.AbstractStepSizeController): Controls
          adaptive stepping.
    """

    func_drift: eqx.Module
    data_size: int
    exact_logp: bool
    t0: float
    dt0: float
    stepsizecontroller: diffrax.AbstractStepSizeController

    def __init__(
        self,
        *,
        data_size,
        exact_logp,
        width_size,
        depth,
        key,
        stepsizecontroller=None,
        func=MLPFunc,
        t0=0.0,
        dt0=1.0,
        **kwargs,
    ):
        if stepsizecontroller is None:
            stepsizecontroller = diffrax.ConstantStepSize()
        keys = jax.random.split(key, 2)
        super().__init__(**kwargs)
        self.func_drift = func(
            data_size=data_size,
            width_size=width_size,
            depth=depth,
            key=keys[0],
        )
        self.data_size = data_size
        self.exact_logp = exact_logp
        self.t0 = t0
        self.dt0 = dt0
        self.stepsizecontroller = stepsizecontroller

    def transform(self, *, y, t1):
        """Transform data through the flow without computing log determinants.

        Args:
            y (jnp.ndarray): Input data points to transform.
            t1 (float): Target time for the transformation.

        Returns:
            jnp.ndarray: Transformed data points.
        """
        term = diffrax.ODETerm(self.func_drift)
        solver = diffrax.ReversibleHeun()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            self.t0,
            t1,
            self.dt0,
            y,
            stepsize_controller=self.stepsizecontroller,
        )
        (y,) = sol.ys
        return y

    def transform_and_log_det(self, *, y, t1, key):
        """Transform data and compute log determinant of the transformation.

        Args:
            y (jnp.ndarray): Input data points to transform.
            t1 (float): Target time for the transformation.
            key (jax.random.key): Random key for stochastic operations.

        Returns:
            tuple: (transformed_y, log_determinant) where transformed_y is the
            transformed data and log_determinant is the change in log
            probability.
        """
        if self.exact_logp:
            term = diffrax.ODETerm(exact_logp_wrapper)
        else:
            term = diffrax.ODETerm(approx_logp_wrapper)
        eps = jax.random.normal(key, y.shape)
        delta_log_likelihood = 0.0

        y = (y, delta_log_likelihood)
        solver = diffrax.ReversibleHeun()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            self.t0,
            t1,
            self.dt0,
            y,
            (eps, self.func_drift),
            stepsize_controller=self.stepsizecontroller,
        )
        (y,), (delta_log_likelihood,) = sol.ys
        return y, delta_log_likelihood

    def inverse_and_log_det(self, *, y, t1, key):
        """Apply inverse transformation and compute the log determinant.

        Args:
            y (jnp.ndarray): Input data points to inverse transform.
            t1 (float): Starting time for the inverse transformation.
            key (jax.random.key): Random key for stochastic operations.

        Returns:
            tuple: (inverse_y, log_determinant) where inverse_y is the inverse
            transformed data and log_determinant is the change in log
            probability.
        """
        if self.exact_logp:
            term = diffrax.ODETerm(exact_logp_wrapper)
        else:
            term = diffrax.ODETerm(approx_logp_wrapper)
        eps = jax.random.normal(key, y.shape)
        delta_log_likelihood = 0.0

        y = (y, delta_log_likelihood)
        solver = diffrax.ReversibleHeun()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t1,
            self.t0,
            -self.dt0,
            y,
            (eps, self.func_drift),
            stepsize_controller=self.stepsizecontroller,
        )
        (y,), (delta_log_likelihood,) = sol.ys

        return y, delta_log_likelihood
