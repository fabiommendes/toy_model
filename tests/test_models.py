import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from toy import Model


class ModelMixin:
    time = 1
    steps = 10

    @pytest.fixture(scope='class')
    def cls(self):
        return self.get_class()

    def test_create_model_class(self):
        self.get_class()

    def get_class(self):
        raise NotImplementedError


class TestBasicModel(ModelMixin):
    def get_class(self):
        class M(Model):
            x = 1, '[1] size'
            D_x = x

        return M

    def test_create_model(self, cls):
        m = cls()
        assert set(m.vars) == {'x'}
        assert m.vars == {'x': m.x}
        assert m.params == {}
        assert m.aux == {}

    def test_override_initial_conditions(self, cls):
        m = cls(x=2)
        assert set(m.values) == {'x'}
        assert m.vars == {'x': m.x}
        assert m.aux == {}
        assert m.params == {}
        assert m.x.value == 2
        assert m.equations == {'x': m.x.symbol}

    def test_model_run(self, cls):
        steps = 10
        m = cls()
        times = np.linspace(0, 1, steps)
        run = m.run(times)
        res = np.exp(times)
        print(run.values)
        print(run.x_ts)
        print(run.x, run.t)
        assert_almost_equal(run.x_ts, res, 3)


class TestParametricModel(ModelMixin):
    def get_class(self):
        class M(Model):
            x = 0, '[1] size'
            k = 1, '[hz] growth rate'
            D_x = k * x

        return M

    def test_create_model(self, cls):
        m = cls()
        assert set(m.values) == {'x', 'k'}
        assert m.vars == {'x': m.x}
        assert m.aux == {}
        assert m.params == {'k': m.k}
        assert m.k.value == 1.0
        assert m.equations == {'x': m.x.symbol}

    def test_override_initial_conditions(self, cls):
        m = cls(x=1, k=2)
        assert set(m.values) == {'x', 'k'}
        assert m.vars == {'x': m.x}
        assert m.aux == {}
        assert m.params == {'k': m.k}
        assert m.k.value == 2
        assert m.x.value == 1
        assert m.equations == {'x': 2 * m.x.symbol}
