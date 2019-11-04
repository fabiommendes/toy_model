import numpy as np

from sidekick import delegate_to
from .meta import Meta
from .model import Model
from ..solvers import Solver
from ..utils import coalesce


class Run:
    """
    This class makes an interface between Models and Solvers.
    """

    solver: Solver
    model: Model

    t = delegate_to('solver')
    state = delegate_to('solver', name='y')
    values = property(lambda self: self._values[:self._idx].T)
    times = property(lambda self: self._times[:self._idx])
    _meta: Meta = delegate_to('model')

    @classmethod
    def from_solver(cls, solver_class, model, **kwargs):
        """
        Create solver from solver class and prepare run method.
        """
        meta = model._meta
        solver = solver_class(meta.diff_fn, y0=meta.y0, t0=meta.t0)
        return cls(solver, model, **kwargs)

    def __init__(self, solver, model, alloc_steps=1, name=None):
        meta = model._meta
        self.name = name
        self.model = model
        self.solver = solver
        self._idx = 1
        self._times = np.ones(alloc_steps, dtype='float64') * float('nan')
        self._values = np.zeros((alloc_steps, meta.vars_size), dtype=model.dtype)
        # self._aux = np.zeros((alloc_steps, meta.aux_size), dtype=model.dtype)
        self._attributes = _make_attributes(self)
        self._times[0] = self.t
        self._values[0] = self.solver.y
        self.solver.callback = self._callback

    def __getattr__(self, attr):
        try:
            attr_getter = self._attributes[attr]
        except KeyError:
            raise AttributeError(attr)
        else:
            return attr_getter()

    def __repr__(self):
        name = f'Run:{self.name}' if self.name else 'Run'
        items = self.var_values().items()
        var_data = ('%s=%r' % item for item in items)
        var_data = ', '.join((f't={self.t}', *var_data))
        return f'<{name} {var_data}>'

    def _callback(self, t, y):
        self._values[self._idx] = y
        self._times[self._idx] = t
        self._idx += 1

    def var_values(self):
        """
        Return a dictionary with the variable values for the current state.
        """
        return self._meta.unvectorize_vars(self.state)

    def run(self, *args, t0=None, tf=None, steps=None, **kwargs):
        """
        Advance simulation in the given time frame.

        Has a similar interface as :meth:`Model.run`.
        """
        if kwargs:
            self.update_ic(**kwargs)

        # Compute the time points
        meta = self.model._meta
        steps = coalesce(steps, meta.steps, 100)
        t0 = coalesce(t0, self.t)
        tf = coalesce(tf, self.t + meta.tf - meta.t0)

        times = times_from_args(*args, start=t0, stop=tf, step=steps)
        self.simulate(times)
        return self

    def simulate(self, times, y0=None):
        """
        Run simulation over the given time points.
        """

        # Fill missing times
        times = np.asarray(times, dtype=float)
        if times.ndim == 0:
            times = times.reshape([1])

        n, m = self._values.shape
        missing = len(times) - (n - self._idx)
        if missing > 0:
            self._values = np.vstack([self._values, np.zeros((missing, m))])
            self._times = np.concatenate([self._times, np.zeros(missing)])

        # Set initial time and value
        if y0 is not None:
            self.solver.y[:] = y0
            self._values[self._idx] = y0
        self._times[self._idx] = times[0]

        self.solver.simulate(times)
        return self

    def step(self, dt):
        """
        Run a single step by
        """
        self.solver.step(dt)
        return self

    def update_ic(self, t0=None, *args, **kwargs):
        """
        Update initial conditions from model. User can override values by
        passing a dictionary positional argument or passing variables as
        positional arguments.
        """
        if t0 is not None:
            self.solver.t = t0
        ic = self.model.var_values(*args, **kwargs)
        self.state[:] = self.model.var_vector(ic)
        self.values[self._idx] = self.y


def times_from_args(*args, start=0, stop=1, step=100):
    """
    Create array of times from arguments to the run() function.
    """

    if len(args) == 0:
        return np.linspace(start, stop, step)
    elif len(args) == 1:
        arg, = args
        try:
            return np.linspace(start, float(arg), step)
        except TypeError:
            return np.asarray(arg)
    elif len(args) == 2:
        start, stop = args
        return np.linspace(start, stop, step)
    elif len(args) == 3:
        start, stop, step = args
        return np.linspace(start, stop, step)
    else:
        raise TypeError('function receive 0 to 3 positional arguments')


def _make_attributes(run: Run):
    """
    Create dictionary mapping attribute names to getter functions
    """

    def make_state_reader(var, data):
        return lambda: meta.read_var(var, data())

    meta = run.model._meta
    attrs = {}

    for var in meta.vars:
        attrs[var] = make_state_reader(var, lambda: run.solver.y)
        attrs[var + '_ts'] = make_state_reader(var, lambda: run.values)

    return attrs
