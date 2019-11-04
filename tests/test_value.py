from sympy import S, symbols
from toy import Value
from toy.core.value import fix_numeric

x, y, z = symbols('x y z')


class TestValue:
    def test_value_replacement(self):
        v1 = Value('a', 42)
        assert v1.replace(x=1).value == 42

        v2 = Value('a', x * y)
        assert v2.replace(x=1).value == y
        assert v2.replace(x=1, y=2).value == 2
        assert v2.replace(x=1, y=2, z=3).value == 2

    def test_fix_numeric(self):
        ns = {
            'x': Value('x', 1),
            'y': Value('y', 2 * x),
            'z': Value('z', 2 * y),
        }
        ns = fix_numeric(ns)
        assert {k: v.value for k, v in ns.items()} == {'x': 1, 'y': 2, 'z': 4}