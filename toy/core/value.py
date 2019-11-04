import operator as op

import numpy as np
from functools import reduce
from sympy import Symbol, Expr
from typing import Optional, Any, Tuple, Mapping, Dict, Set, Union

from sidekick import import_later, Record
from toy.utils import substitute
from ..unit import DIMENSIONLESS
from ..utils import is_numeric

expr = import_later('.expr', package=__name__)
NumericType = Union[int, float, np.ndarray]
ValueType = Union[NumericType, Any]


class Value(Record):
    """
    Represents a numeric value.
    """

    # Record fields
    name: str
    value: ValueType
    symbol: Symbol
    unit: object = DIMENSIONLESS
    description: str = ''
    lower: Optional[ValueType] = None
    upper: Optional[ValueType] = None
    shape: Optional[Tuple[int, ...]] = None

    # Properties
    is_numeric = property(lambda self: is_numeric(self.value))
    size = property(lambda self: reduce(op.mul, self.shape, 1))

    def __init__(self, name: str, value: ValueType, *, shape=None, **kwargs):
        if shape is None and value is None:
            kwargs['shape'] = (1,)
        elif shape is None:
            kwargs['shape'] = getattr(value, 'shape', (1,))
        else:
            vshape = getattr(value, 'shape', (1,))
            assert tuple(shape) == vshape
            kwargs['shape'] = vshape
        symbol = kwargs.pop('symbol', Symbol(name, real=True))
        super().__init__(name, value, symbol, **kwargs)

    def __repr__(self):
        return 'Value(%r, %r)' % (self.name, self.value)

    def __hash__(self):
        return id(self)

    def __gt__(self, other):
        if isinstance(other, Value):
            return self.name > other.name
        elif other is None:
            return True
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Value):
            return self.name < other.name
        elif other is None:
            return False
        return NotImplemented

    def copy(self, **kwargs):
        """
        Create a copy, possibly overriding some attribute.
        """
        kwargs = {
            'name': self.name,
            'value': self.value,
            'shape': self.shape,
            'symbol': self.symbol,
            'description': self.description,
            'unit': self.unit,
            'lower': self.lower,
            'upper': self.upper,
            **kwargs,
        }
        return Value(**kwargs)

    def replace(self, **kwargs) -> 'Value':
        """
        Return a new Value that replaces the dependent variables by the ascribed
        values.
        """
        x = self.value

        if self.name in kwargs:
            value = kwargs[self.name]
        elif is_numeric(x):
            value = x
        elif isinstance(x, Expr):
            value = substitute(x, kwargs)
        else:
            raise NotImplementedError(x)

        return self.copy(value=value)

    def dependent_variables(self) -> Set[str]:
        """
        A set with all dependent variable names.
        """
        x = self.value

        if is_numeric(x):
            return set()
        elif isinstance(x, Expr):
            return {str(x) for x in x.atoms()}
        elif callable(x):
            raise NotImplementedError(x)


def replace_values(substitutions: Mapping[str, Any], ns: Dict[str, Value]):
    """
    Replace values given in dictionary into declarations.

    Args:
        substitutions:
            A mapping from variable names to their corresponding numerical
            or expression values.
        ns:
            A mapping from variable names to their corresponding Value
            declarations.
    """
    return {k: v.replace(**substitutions) for k, v in ns.items()}


def fix_numeric(ns: Mapping[str, Value]) -> Dict[str, Value]:
    """
    Fix the values of all numeric variables in namespace recursively.
    """
    numeric = {}
    subs = {}
    ns = dict(ns)
    size = None

    while len(ns) != size:
        size = len(ns)
        for k, v in list(ns.items()):
            if v.is_numeric:
                numeric[k] = ns.pop(k)
                subs[k] = v.value
            else:
                ns[k] = v.replace(**subs)

    return {**ns, **numeric}
