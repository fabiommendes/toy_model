import numpy as np
from sympy import Expr
from typing import Mapping, Any, Dict

import sidekick as sk
from toy.solvers import SOLVERS
from .meta import Meta
from .model_meta import ModelMeta
from .value import Value, fix_numeric, NumericType
from ..utils import substitute, coalesce

run = sk.import_later('..run', package=__name__)


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
        initial_conditions = dict(ic, **kwargs)

        # Initialize vars, params, aux
        subs = initial_conditions
        values = {k: v.replace(**subs) for k, v in self.values.items()}
        self.vars = {k: v for k, v in values.items() if k in self.equations}

        # We now must decide what is parameter and what is not
        values = {k: v for k, v in values.items() if k not in self.vars}
        values = fix_numeric(values)
        self.params = {k: v for k, v in values.items() if v.is_numeric}
        self.aux = {k: v for k, v in values.items() if k not in self.params}

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
        self.values = {**self.vars, **self.params, **self.aux}
        for k, v in self.values.items():
            setattr(self, k, v)

    def run(self, *args, solver='rk4', t0=None, tf=None, steps=None, name=None, **kwargs) -> 'Run':
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
        meta = self._meta
        steps = coalesce(steps, meta.steps, 100)
        t0 = coalesce(t0, meta.t0)
        tf = coalesce(tf, meta.tf)
        runner = self.runner(solver, name=name)
        times = run.times_from_args(*args, start=t0, stop=tf, step=steps)
        return runner.run(times, **kwargs)

    def runner(self, solver='rk4', **kwargs):
        """
        Return a run instance, without running simulation.
        """
        if isinstance(solver, str):
            solver = SOLVERS[solver]
            return run.Run.from_solver(solver, self, **kwargs)
        else:
            return run.Run(solver, self, **kwargs)

    def var_values(*args, **kwargs) -> Dict[str, NumericType]:
        """
        Return a dictionary with initial conditions for the dynamic variables.

        It can override initial conditions by passing them in a mapping as
        the first positional argument or as keyword arguments.
        """
        self, ns = extract_ns(args, kwargs)
        initial = {k: v.value for k, v in self.vars.items()}
        initial.update(ns)
        return initial

    def aux_values(*args, **kwargs) -> Dict[str, NumericType]:
        """
        Return a dictionary with initial conditions for the computed
        variables.

        It can override initial conditions by passing them in a mapping as
        the first positional argument or as keyword arguments.
        """
        self, ns = extract_ns(args, kwargs)
        values = {k: v.replace(**ns) for k, v in self.aux.items()}
        return fix_numeric(values)

    def param_values(self, *args, **kwargs) -> Dict[str, NumericType]:
        """
        Analogous to :meth:`initial_computed` and :meth:`initial_vars`, but
        return parameters. Since parameters are static, arguments have no
        effect.
        """
        return {k: v.value for k, v in self.params.items()}

    def var_vector(self, values: Mapping[str, float]):
        """
        Convert dictionary of dynamic variables into an array.
        """
        return self._meta.compiler.vectorize_vars(self.var_values(values))

    def aux_vector(self, values: Mapping[str, float]):
        """
        Convert dictionary of auxiliary variables into an array.
        """
        return self._meta.compiler.vectorize_aux(self.aux_values(values))


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
