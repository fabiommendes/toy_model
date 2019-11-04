from . import definitions
from .unit import parse_unit, parse_unit_msg, DIMENSIONLESS, UNITS

for k, v in vars(definitions).items():
    if not k.startswith('_'):
        setattr(UNITS, k, v)
del k, v