from numbers import Number

import numpy as np
from sympy import Expr, S
from typing import Mapping, Any, Dict

from ..compiler import Compiler
from .meta import Meta
from .model_meta import ModelMeta
from .value import Value, fix_numeric, NumericType
from ..utils import substitute

class Model(metaclass=ModelMeta):
    """
    A Model is a declaration of a system of differential equations.
    """

    #: Type of elements in equation. Toy model only accepts uniformly typed
    #: values. Choose something like np.float64 or np.float32
    dtype: type = np.float64

    #: Map variable names to value declarations
    values: Mapping[str, Value]

    #: Map variable names to their corresponding dynamic equation
    equations: Mapping[str, Any]

    def __init__(self, ic=(), **kwargs):
        self._meta = Meta(self)
        self.initial_conditions = dict(ic, **kwargs)

        # Initialize vars, params, computed_terms
        subs = self.initial_conditions
        values = {k: v.replace(**subs) for k, v in self.values.items()}
        self.vars = {k: v for k, v in values.items() if k in self.equations}

        # We now must decide what is parameter and what is not
        values = {k: v for k, v in values.items() if k not in self.vars}
        values = fix_numeric(values)
        self.params = {k: v for k, v in values.items() if v.is_numeric}
        self.computed_terms = {k: v for k, v in values.items() if k not in self.params}

        # Replace values in equations
        params = {k: v.value for k, v in self.params.items()}
        if params:
            eqs = {}
            for k, eq in self.equations.items():
                if isinstance(eq, Expr):
                    eq = substitute(eq, params)
                elif callable(eq):
                    raise NotImplementedError(eq)
                eqs[k] = eq
            self.equations = eqs

        # Save all values as attributes
        self.values = {**self.vars, **self.params, **self.computed_terms}
        for k, v in self.values.items():
            setattr(self, k, v)

        # Obtain the derivative and compute functions
        self._compiler = Compiler(self.vars, self.computed_terms, self.equations)

    def run(self, *args, solver='euler', **kwargs):
        """
        Run simulation and return a Run object.

        Examples:
            run(t) -> run simulation from time initial time to to t
            run(t0, tf) -> run simulation from time t0 to tf
            run(t0, tf, steps) -> ditto, but control the number of steps
            run([t1, t2, ...]) -> run simulation through given times

        Keyword arguments:
            solver:
                Method used to solve equation:
                    - 'euler'
                    - 'rk4'
        """
        x0 = self.initial_state(kwargs)
        storage_size = len(x0) + len(self.initial_computed())

        diff = self._compiler.compile_diff_fn(require_computed=True)
        compute = self._compiler.compile_computed_terms_fn()

        times = times_from_args(*args)
        ts_data = np.ndarray((len(times), storage_size), dtype=self.dtype)
        ts_data[0, :len(x0)] = x0
        ts_data[0, len(x0):] = compute(x0, times[0])

        euler_run(x0, times, compute, diff, ts_data)
        return np.asarray(ts_data.T, dtype=self.dtype)

    def initial_vars(*args, **kwargs) -> Dict[str, NumericType]:
        """
        Return a dictionary with initial conditions for the dynamic variables.

        It can override initial conditions by passing them in a mapping as
        the first positional argument or as keyword arguments.
        """
        self, ns = extract_ns(args, kwargs)
        return ns

    def initial_computed(*args, **kwargs) -> Dict[str, NumericType]:
        """
        Return a dictionary with initial conditions for the computed
        variables.

        It can override initial conditions by passing them in a mapping as
        the first positional argument or as keyword arguments.
        """
        self, ns = extract_ns(args, kwargs)
        values = {k: v.replace(**ns) for k, v in self.computed_terms.items()}
        return fix_numeric(values)

    def initial_params(self, *args, **kwargs) -> Dict[str, NumericType]:
        """
        Analogous to :meth:`initial_computed` and :meth:`initial_vars`, but
        return parameters. Since parameters are static, arguments have no
        effect.
        """
        return {k: v.value for k, v in self.params.items()}

    def initial_state(self, *args, **kwargs) -> np.ndarray:
        """
        Similar to :meth:`initial_vars`, but return an ndarray with the initial
        state.
        """
        x0 = self.initial_vars(*args, **kwargs)
        return self._compiler.initial_state(x0)


def euler_run(x, times, compute, diff, storage):
    t = times[0]
    time_deltas = times[1:] - times[:-1]

    for i, dt in enumerate(time_deltas, start=1):
        y = compute(x, t)
        x += dt * diff(y, x, t)
        storage[i] = [*x, *y]
        t += dt

    return x


def times_from_args(*args):
    if len(args) == 0:
        return np.linspace(0, 1)
    elif len(args) in (2, 3):
        return np.linspace(*args)
    else:
        arg, = args
        if isinstance(arg, Number):
            return np.linspace(0, 1)
        return np.asarray(arg)


def extract_ns(args, kwargs):
    self, *args = args
    if args:
        ns, = args
        ns = {**ns, **kwargs}
    else:
        ns = kwargs

    invalid = set(ns) - set(self.vars)
    if invalid:
        raise TypeError(f'cannot set variables: {invalid}')
    return self, ns
