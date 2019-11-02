from sympy import S, symbols
from toy import Value
x, y = symbols('x y')


class TestValue:
    def test_value_replacement(self):
        v1 = Value('a', 42)
        assert v1.replace(x=1).value == 42

        v2 = Value('a', x * y)
        assert v2.replace(x=1).value == y
        assert v2.replace(x=1, y=2).value == 2
        assert v2.replace(x=1, y=2, z=3).value == 2
