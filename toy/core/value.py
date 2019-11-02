import operator as op

from functools import reduce
from sympy import Symbol
import numpy as np
from typing import Optional, Any, Tuple, Mapping, Dict, Set, Union

from sidekick import import_later, Record
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
        symbol = kwargs.pop('symbol', Symbol(name))
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

    def replace(self, **kwargs) -> 'Value':
        """
        Return a new Value that replaces the dependent variables by the ascribed
        values.
        """
        if self.is_numeric:
            return self
        raise NotImplementedError

    def dependent_variables(self) -> Set[str]:
        """
        Set of all dependent variable names.
        """
        if self.is_numeric:
            return set()
        raise NotImplementedError


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
    return dict(ns)
