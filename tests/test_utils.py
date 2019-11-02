import numpy as np
from sympy import Symbol, Integer, Float, S

from toy.utils import is_numeric


def test_is_numeric():
    for number in [1, 2.0, 3j, Integer(1), Float(1.0), S(1), S(1)/2, np.array(1.0)]:
        assert is_numeric(number) is True

    x = Symbol('x')
    for non_number in ["foo", np.array("foo"), x, x + 1, lambda x: x]:
        assert is_numeric(non_number) is False
