from sympy.physics.units import meter, second

from toy.unit import parse_unit, DIMENSIONLESS


class TestUnits:
    def test_parse_unit(self):
        assert parse_unit('1') == DIMENSIONLESS
        assert parse_unit('m/s') == meter / second
        assert parse_unit('m/s2') == meter / second ** 2
        assert parse_unit('m s-2') == meter / second ** 2
        assert parse_unit('m s') == meter * second
        assert parse_unit('m * s') == meter * second
