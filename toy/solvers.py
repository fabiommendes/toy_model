import numpy as np
from functools import partial
from typing import Iterable, Tuple

ST = np.ndarray
T = np.ndarray


class Solver:
    """
    A solver class describes an algorithm that solves some particular
    ODE problem. The basic solver expects a ODE written as bellow,

        dy/dt = fn(t, y);  y(t0) = y0;

    User must provide the initial conditions t0, y0, as well as the derivative
    function fn(t, y).

    The solver may track statistics about execution and is responsible for
    evolving its current state and time variable.
    """
    __slots__ = ('fn', 'y', 't', 'callback', 'ncalls', 'niter')

    def __init__(self, fn, y0: ST, t0=0.0, callback=None, log=True):
        self.fn = fn
        self.y = np.asarray(y0) + np.array([0.0])
        self.t = t0 + 0.0
        self.callback = callback
        self.ncalls = 0
        self.niter = 0

        if log:
            self.fn = self._logging_fn(fn)
            self.callback = self._logging_callback(callback)

    def call(self, func):
        """
        Call function func(t, y) with the current state of simulation.
        """
        func(self.t, self.y)
        return self

    def step_function(self, t, y, dt) -> ST:
        """
        Resolve one step of simulation starting at time t and state x and return
        the solution after time increment dt.

        This function *does not* change the state of simulation and does not call
        the callback function. It can be safely called multiple times to test
        different steps sizes, and initial conditions.
        """
        raise NotImplementedError

    def step(self, dt) -> 'Solver':
        """
        Evolve the current state by a single time delta.

        Return solver, which makes it usable in a fluent interface.
        """
        self.y[:] = self.step_function(self.t, self.y, dt)
        self.t += dt
        cb = self.callback
        if cb is not None:
            cb(self.t, self.y)
        return self

    def steps(self, dt: T) -> 'Solver':
        """
        Like :meth:`step', but advance by multiple steps.

        Return solver, which makes it usable in a fluent interface.
        """
        for dt in dt:
            self.step(dt)
        return self

    def solve_steps(self, dt) -> np.ndarray:
        """
        Like :meth:`steps', but return an array with all results of simulation.

        The resulting array **includes** the initial state.
        """

        data = np.zeros((len(dt) + 1, len(self.y)), dtype=self.y.dtype)
        data[0] = self.y
        cb = self.callback
        idx = 1

        try:
            def callback(t, y):
                nonlocal idx
                if cb is not None:
                    cb(t, y)
                data[idx] = y
                idx += 1

            self.callback = callback
            self.steps(dt)
        finally:
            self.callback = cb
        return data

    def simulate(self, times, y0=None) -> 'Solver':
        """
        Simulate ODE over for all given times.

        This method resets the starting time and state. It returns the solver, which
        makes it usable as a fluent interface.
        """

        times = np.asarray(times, dtype=float)
        dt = times[1:] - times[:-1]
        self.t = times[0]
        if y0 is not None:
            self.y[:] = y0
        self.steps(dt)
        return self

    def solve(self, times, y0=None):
        """
        Simulate system, and return an array with the resulting states.
        """
        if y0 is not None:
            self.y[:] = y0
        times = np.asarray(times)
        self.t = times[0]
        return self.solve_steps(times[1:] - times[:-1])

    def iter_steps(self, dt, first=False) -> Iterable[Tuple[float, ST]]:
        """
        Return in iterator over the state of simulation by stepping over all
        given time deltas.

        Times can be a lazy sequence and even infinite.
        """
        if first:
            yield self.t, self.y.copy()

        for dt in dt:
            y = self.step(dt).y.copy()
            yield self.t, y

    def iter_times(self, times, y0=None) -> Iterable[Tuple[float, ST]]:
        """
        Return in iterator over the state of simulation at all given time points.

        Times can be a lazy sequence and even infinite.
        """
        times = iter(times)
        if y0 is not None:
            self.y[:] = y0
        self.t = next(times) + 0.0
        yield self.t, self.y.copy()

        for t in times:
            y = self.step(t - self.t).y.copy()
            yield self.t, y

    def clear_logs(self):
        """
        Reset all current logs.
        """
        self.ncalls = 0
        self.niter = 0

    def _logging_fn(self, fn):
        def log_fn(t, y):
            self.ncalls += 1
            return fn(t, y)

        return log_fn

    def _logging_callback(self, callback):
        if callback is None:
            return None

        def log_callback(t, y):
            self.niter += 1
            return callback(t, y)

        return log_callback


class Euler(Solver):
    """
    Explicit Euler method.
    """

    __slots__ = ()

    def step_function(self, t, x, dt):
        return x + dt * self.fn(t, x)


class RK2(Solver):
    """
    Second order Runge-Kutta method.

    Parametrizes a series of methods:

        alpha = 1/2 - Midpoint rule
        alpha = 2/3 - Ralston's rule
        alpha = 1   - Heun/trapezoid rule
    """
    __slots__ = ('alpha', 'w1', 'w2')

    def __init__(self, *args, alpha=0.5, **kwargs):
        self.alpha = alpha
        self.w1 = 1 - 0.5 / alpha
        self.w2 = 1 - self.w1
        super().__init__(*args, **kwargs)

    def step_function(self, t, x, dt):
        diff = self.fn
        tau = self.alpha * dt
        k1 = diff(t, x)
        k2 = diff(t + tau, x + k1 * tau)
        return x + dt * (self.w1 * k1 + self.w2 * k2)


class RK4(Solver):
    """
    Classic fourth order Runge-Kutta method.
    """
    __slots__ = ()

    def step_function(self, t, x, dt):
        diff = self.fn
        tau = dt / 2
        k1 = diff(t, x)
        k2 = diff(t + tau, x + k1 * tau)
        k3 = diff(t + tau, x + k2 * tau)
        k4 = diff(t + dt, x + k3 * dt)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def compute_to_diff_cb(size, compute, diff_y, cb):
    y = None

    def diff(x, t):
        nonlocal y
        y_ = compute(x, t)
        if y is None:
            y = y_
        return diff_y(y, x, t)

    def callback(x):
        nonlocal y
        cb(x, y)
        y = None

    return diff, callback


def store_data_cb(storage, mask=slice(None, None)):
    def store_data_coro():
        yield
        i = 0
        while True:
            storage[i + 1] = (yield)[mask]
            i += 1

    coro = store_data_coro()
    next(coro)
    return coro.send


def store_all_data_cb(storage, mask_vars=slice(None, None),
                      mask_computed=slice(None, None)):
    def store_data_coro():
        yield
        i = 0
        while True:
            x, y = yield
            storage[i + 1, 0:len(x)] = x[mask_vars]
            storage[i, 0:len(x)] = x[mask_computed]
            i += 1

    coro = store_data_coro()
    next(coro)
    return coro.send


SOLVERS = {
    'euler': Euler,
    'midpoint': RK2,
    'rk2': RK2,
    'ralston': partial(RK2, alpha=2 / 3),
    'heun': partial(RK2, alpha=1.0),
    'rk4': RK4,
}
