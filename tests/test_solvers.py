import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from toy.solvers import Euler, RK2, RK4


class TestSteppedSolvers:
    @pytest.fixture
    def euler(self):
        return Euler(lambda t, y: y, [1.0])

    def test_euler_solver_step_function(self, euler):
        assert euler.y == 1.0
        assert euler.step(1.0).y == 2.0
        assert euler.step(1.0).y == 4.0

    def test_euler_solution(self, euler):
        assert euler.steps([1, 1, 1]).y == 8.0

    def test_euler_simulation_times(self, euler):
        assert euler.simulate([0, 1, 2, 3], [1]).y == 8.0

    def test_euler_trajectory(self, euler):
        ys = euler.solve([0, 1, 2, 3, 4], euler.y)
        assert_almost_equal(ys[:, 0], [1, 2, 4, 8, 16])

    def test_precision_of_different_methods(self):
        times = np.linspace(0, 1, 5)
        args = lambda t, y: y, [1.0]
        exact = np.exp(times)
        err = lambda x: np.abs(x[:, 0] - exact).sum()

        euler = Euler(*args).solve(times)
        rk2 = RK2(*args).solve(times)
        rk4 = RK4(*args).solve(times)
        e_euler = err(euler)
        e_rk2 = err(rk2)
        e_rk4 = err(rk4)
        assert e_euler > e_rk2 > e_rk4 < 1e-3
