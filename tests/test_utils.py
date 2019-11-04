import numpy as np
from sympy import Symbol, Integer, Float, S

from toy.utils import is_numeric, coalesce


def test_is_numeric():
    for number in [1, 2.0, 3j, Integer(1), Float(1.0), S(1), S(1) / 2, np.array(1.0)]:
        assert is_numeric(number) is True

    x = Symbol('x')
    for non_number in ["foo", np.array("foo"), x, x + 1, lambda x: x]:
        assert is_numeric(non_number) is False


def test_coalesce():
    assert coalesce(None, None) is None
    assert coalesce(None, 1) == 1
    assert coalesce(0, None, 1) == 0
    assert coalesce(None, 1, None) == 1
