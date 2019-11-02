from numbers import Number

import numpy as np
from sympy import Number as SymbNumber

NUMBER_TYPES = (int, float, SymbNumber, Number)
NUMPY_NON_TYPES = {np.str_, np.object_}


def is_numeric(x):
    """
    Test if x is of an explicit numeric type.

    Symbols/variables are not considered to be numeric, but sympy's arbitrary
    precision numbers or numeric constants like sympy.pi are.
    """
    if isinstance(x, NUMBER_TYPES):
        return True
    elif isinstance(x, np.ndarray):
        return x.dtype.type not in NUMPY_NON_TYPES
    return False


def as_dict(x):
    """
    Coerce argument to dictionary.
    """
    if x is None:
        return {}
    elif isinstance(x, dict):
        return x
    else:
        return dict(x)
