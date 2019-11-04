from sympy import S
from sympy.physics.units import Quantity, mass, kg, year, Dimension

_factor = {'': 1, 'k': 1000, 'M': 1_000_000, 'G': 1_000_000_000}
_names = {'': '', 'k': 'kilo', 'M': 'mega_', 'G': 'giga_'}
One = S.One


def scale(base, kind, prefixes="", multiplier=1):
    """
    Define a scale of similar quantities.
    """
    base, abbrev = base.split('/')

    res = []
    for prefix in ['', *prefixes]:
        q = Quantity(_names[prefix] + base, abbrev=abbrev)
        q.set_dimension(kind[0])
        q.set_scale_factor(multiplier * _factor[prefix] * kind[1])
        res.append(q)

    return res if prefixes else res[0]


#
# Aliases
#
yr = yrs = year

#
# Climatic units
#
tC, MtC, GtC = scale("tons_of_carbon/tC", (mass, kg), "MG", 1000)

#
# Economic units
#
money = Dimension(name="mass", symbol="M")
globals()['U$'] = dollar = dollars = scale('dollars/U$', (money, One))

#
# Clean up
#
del Quantity, Dimension, S, scale
