from numbers import Number

import numpy as np
from sympy import Number as SymbNumber

NUMBER_TYPES = (int, float, SymbNumber, Number)
NUMPY_NON_TYPES = {np.str_, np.object_}


def coalesce(*args):
    """
    Return the first non-null argument.
    """
    for arg in args:
        if arg is not None:
            return arg


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


def substitute(expr, vars):
    """
    Substitute all variables into expression.

    Args:
        expr:
            A sympy expresssion
        vars:
            A mapping from variable names to the corresponding substitution
            values.
    """
    subs = {}
    for atom in expr.atoms():
        name = str(atom)
        try:
            subs[atom] = vars[name]
        except KeyError:
            pass

    value = expr.subs(subs).evalf()
    if isinstance(value, SymbNumber) and value == int(value):
        return int(value)
    return value
