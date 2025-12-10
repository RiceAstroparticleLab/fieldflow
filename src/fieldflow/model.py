"""Neural network models for continuous normalizing flows.

This module defines the neural network architectures used to model electric
field distortions in dual-phase TPCs. The CNF uses separate networks for
the extraction field (z-independent) and drift field (z-dependent).

Two approaches are provided for enforcing Maxwell's equations:

- **MLPFunc**: Direct vector field parameterization. Requires explicit curl
  penalty during training to enforce curl-free constraint.
- **DriftFromPotential**: Scalar potential parameterization where the drift
  is the negative gradient of the potential. Curl-free by construction.
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
    """Multilayer perceptron that directly models the drift vector field.

    This network takes (x, y, t) as input and outputs a 2D drift vector.
    When used for electric field modeling, training should include a curl
    penalty to encourage physically valid (curl-free) fields.

    Attributes:
        layers: List of linear layers forming the MLP.
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


class ScalarMLPFunc(eqx.Module):
    """Multilayer perceptron that models a scalar potential field.

    This network takes (x, y, t) as input and outputs a scalar value
    representing the potential at that point. Used by DriftFromPotential
    to derive curl-free drift fields via automatic differentiation.

    Attributes:
        layers: List of linear layers forming the MLP.
    """

    layers: list[eqx.nn.Linear]

    def __init__(self, *, data_size, width_size, depth, key, **kwargs):
        """Initialize the MLP scalar potential function.

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
        # modify final output to be scalar
        if depth == 0:
            layers.append(eqx.nn.Linear(data_size + 1, 1, key=keys[0]))
        else:
            layers.append(
                eqx.nn.Linear(data_size + 1, width_size, key=keys[0])
            )
            for i in range(depth - 1):
                layers.append(
                    eqx.nn.Linear(width_size, width_size, key=keys[i + 1])
                )

            layers.append(eqx.nn.Linear(width_size, 1, key=keys[-1]))
        self.layers = layers

    def __call__(self, t, y, args):  # noqa: ARG002
        """Compute the scalar potential at given time and state.

        Args:
            t (float): Current time.
            y (jnp.ndarray): Current state vector.

        Returns:
            jnp.ndarray: Potential as a vector of shape (batch_size, 1).
        """
        # Ensure t is a 1D array and broadcast it to match y's shape
        t = jnp.broadcast_to(jnp.asarray(t), (*y.shape[:-1], 1))
        y = jnp.concatenate((y, t), axis=-1)

        for layer in self.layers[:-1]:
            y = layer(y)
            y = jax.nn.silu(y)
        y = self.layers[-1](y)
        # return scalar potential value
        return jnp.squeeze(y, axis=-1)

class DriftFromPotential(eqx.Module):
    """Drift function derived from a scalar potential.

    This class wraps a ScalarMLPFunc and computes the drift as the negative
    gradient of the learned potential. This guarantees curl-free fields by
    construction, satisfying Maxwell's equations without explicit penalties.

    Attributes:
        model: The underlying scalar potential network.
    """

    model: ScalarMLPFunc

    def __init__(self, *, data_size, width_size, depth, key, **kwargs):
        """Initialize the drift-from-potential model.

        Args:
            data_size: Dimensionality of the spatial coordinates.
            width_size: Width of hidden layers in the potential network.
            depth: Number of hidden layers in the potential network.
            key: JAX random key for parameter initialization.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        # Initialize the scalar potential model
        self.model = ScalarMLPFunc(
            data_size=data_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(self, t, y, args):
        """Compute drift as negative gradient of the scalar potential.

        Args:
            t: Current time (z coordinate in physical terms).
            y: Current spatial position (x, y).
            args: Additional arguments (unused, for interface compatibility).

        Returns:
            Drift vector pointing in direction of steepest potential descent.
        """
        gradient = jax.grad(lambda y: self.scalar_pot(t, y, args))(y)
        return -gradient

    def scalar_pot(self, t, y, args):
        """Evaluate the scalar potential at a point.

        Args:
            t: Current time.
            y: Spatial position.
            args: Additional arguments passed to the model.

        Returns:
            Scalar potential value (summed over batch if applicable).
        """
        return self.model(t, y, args).sum()

class ContinuousNormalizingFlow(eqx.Module):
    """Continuous normalizing flow for dual-phase TPC field modeling.

    This model uses neural ODEs to learn electric field distortions. The
    architecture mirrors dual-phase TPC physics with two sequential flows:

    1. **Extraction phase** (func_extract): Models z-independent field
       distortions that affect the (x, y) distribution uniformly regardless
       of drift distance.
    2. **Drift phase** (func_drift): Models z-dependent field distortions
       where the effect accumulates with drift distance.

    Attributes:
        func_drift: Neural network for the z-dependent drift field.
        func_extract: Neural network for the z-independent extraction field.
        data_size: Dimensionality of the spatial data (typically 2 for x, y).
        exact_logp: If True, compute exact Jacobian trace. If False, use
            Hutchinson estimator.
        t0: Initial time for ODE integration.
        dt0: Initial step size for ODE solver.
        extract_t1: Integration end time for extraction phase.
        stepsizecontroller: Adaptive step size controller for ODE solver.
    """

    func_drift: eqx.Module
    data_size: int
    exact_logp: bool
    t0: float
    dt0: float
    stepsizecontroller: diffrax.AbstractStepSizeController

    func_extract: eqx.Module
    extract_t1: float

    def __init__(
        self,
        *,
        data_size,
        exact_logp,
        width_size,
        depth,
        key,
        stepsizecontroller=None,
        func=ScalarMLPFunc, ### DEFAULT, should not be
        t0=0.0,
        dt0=1.0,
        extract_t1 = 10,
        **kwargs,
    ):
        """Initialize the continuous normalizing flow.

        Creates two neural networks with identical architecture: one for the
        extraction phase and one for the drift phase.

        Args:
            data_size: Dimensionality of spatial coordinates.
            exact_logp: Whether to use exact Jacobian trace computation.
            width_size: Width of hidden layers in drift/extraction networks.
            depth: Number of hidden layers in drift/extraction networks.
            key: JAX random key for parameter initialization.
            stepsizecontroller: ODE step size controller. Defaults to
                ConstantStepSize if not provided.
            func: Neural network class for drift/extraction functions.
                Use MLPFunc for vector field or DriftFromPotential for
                scalar potential approach.
            t0: Initial time for ODE integration.
            dt0: Initial step size for ODE solver.
            extract_t1: End time for extraction phase integration.
            **kwargs: Additional arguments passed to parent class.
        """
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
        self.func_extract = (
            func(
                data_size=data_size,
                width_size=width_size,
                depth=depth,
                key=keys[1],
            )
        )
        self.data_size = data_size
        self.exact_logp = exact_logp
        self.t0 = t0
        self.dt0 = dt0
        self.stepsizecontroller = stepsizecontroller
        self.extract_t1 = extract_t1

    def transform(self, *, y, t1):
        """Transform coordinates through extraction and drift phases.

        Applies the full field distortion model without tracking probability
        changes. First applies the extraction field (z-independent), then
        the drift field (z-dependent, integrated to time t1).

        Args:
            y: Input spatial coordinates of shape (data_size,).
            t1: Target time for drift phase (corresponds to scaled z depth).

        Returns:
            Transformed coordinates after both flow phases.
        """
        term = diffrax.ODETerm(self.func_extract)
        solver = diffrax.Euler()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            self.t0,
            self.extract_t1,
            self.dt0,
            y,
            stepsize_controller=self.stepsizecontroller,
        )
        (y,) = sol.ys

        term = diffrax.ODETerm(self.func_drift)
        solver = diffrax.Euler()
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
        """Transform coordinates and compute the log determinant Jacobian.

        Applies extraction then drift phases while accumulating the log
        determinant of the transformation Jacobian, needed for density
        estimation via the change of variables formula.

        Args:
            y: Input spatial coordinates of shape (data_size,).
            t1: Target time for drift phase (scaled z coordinate).
            key: JAX random key (used for Hutchinson estimator if exact_logp
                is False).

        Returns:
            Tuple of (transformed_y, log_det) where log_det is the accumulated
            log determinant from both phases.
        """
        if self.exact_logp:
            term = diffrax.ODETerm(exact_logp_wrapper)
        else:
            term = diffrax.ODETerm(approx_logp_wrapper)
        eps = jax.random.normal(key, y.shape)
        delta_log_likelihood = 0.0

        y = (y, delta_log_likelihood)
        solver = diffrax.Euler()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            self.t0,
            self.extract_t1,
            self.dt0,
            y,
            (eps, self.func_extract),
            stepsize_controller=self.stepsizecontroller,
            max_steps=16384,
        )
        (y,), (delta_log_likelihood,) = sol.ys

        y = (y, delta_log_likelihood)
        solver = diffrax.Euler()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            self.t0,
            t1,
            self.dt0,
            y,
            (eps, self.func_drift),
            stepsize_controller=self.stepsizecontroller,
            max_steps =16384,
        )
        (y,), (delta_log_likelihood,) = sol.ys
        return y, delta_log_likelihood

    def inverse_and_log_det(self, *, y, t1, key):
        """Apply inverse transformation and compute the log determinant.

        Reverses the flow transformation: first inverts the drift phase
        (from t1 back to t0), then inverts the extraction phase.

        Args:
            y: Transformed coordinates to invert.
            t1: Time to invert from for drift phase (scaled z coordinate).
            key: JAX random key for Hutchinson estimator if needed.

        Returns:
            Tuple of (original_y, log_det) where original_y is the recovered
            input coordinates and log_det is the accumulated log determinant.
        """
        if self.exact_logp:
            term = diffrax.ODETerm(exact_logp_wrapper)
        else:
            term = diffrax.ODETerm(approx_logp_wrapper)
        eps = jax.random.normal(key, y.shape)
        delta_log_likelihood = 0.0

        y = (y, delta_log_likelihood)
        solver = diffrax.Euler()
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

        y = (y, delta_log_likelihood)
        solver = diffrax.Euler()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            self.extract_t1,
            self.t0,
            -self.dt0,
            y,
            (eps, self.func_extract),
            stepsize_controller=self.stepsizecontroller
        )
        (y,), (delta_log_likelihood,) = sol.ys

        return y, delta_log_likelihood
