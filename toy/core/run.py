import numpy as np


class Run:
    def __init__(self, data, map):
        pass

    def __getattr__(self, attr):
        return []


class Runner:
    def __init__(self, ic, diff):
        self.initial_condition = ic
        self.diff_func = diff

    def solve(self, times):
        raise NotImplementedError


class EulerRunner(Runner):
    def step(self, dt):
        pass

    def solve(self, times):
        state = self.initial_condition
        diff = self.diff_func
        t = times[0]
        time_deltas = times[1:] - times[:-1]
        states = [state]

        for dt in time_deltas:
            self.step(dt)
            state = state + dt * diff(state, t)
            states.append(state)
            t += dt

        return Run(np.hstack(states))
