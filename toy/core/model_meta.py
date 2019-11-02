from numbers import Number

import numpy as np
import sympy
from collections.abc import Mapping
from sympy import Symbol
from sympy.core.relational import Relational

from toy.unit import DIMENSIONLESS, parse_unit_msg
from toy.utils import as_dict, is_numeric
from toy.core.value import Value

base_model = None


class ModelMeta(type):
    """
    Metaclass for all Model subclasses.
    """

    @classmethod
    def __prepare__(mcs, name, bases):
        if base_model is None:
            return {}
        return Environment()

    def __new__(mcs, name, bases, ns):
        global base_model

        if base_model is None:
            base_model = type.__new__(mcs, name, bases, ns)
            return base_model
        else:
            return type.__new__(mcs, name, bases, ns.finalize())


class Environment(Mapping):
    """
    A mapping that interprets a Model class creation.
    """

    def __init__(self, ns=None, values=None, equations=None, invariants=None):
        self.namespace = as_dict(ns)
        self.values = as_dict(values)
        self.equations = as_dict(equations)
        self.invariants = as_dict(invariants)
        self.symbols = {'t'}
        self.namespace['t'] = Symbol('t')
        self.namespace['values'] = self.values
        self.namespace['equations'] = self.equations
        self.namespace['invariants'] = self.invariants
        self.lower = {}
        self.upper = {}

    def __iter__(self):
        yield from self.namespace

    def __getitem__(self, key):
        return self.namespace[key]

    def __len__(self):
        return len(self.namespace)

    def __setitem__(self, k, v):
        if k.startswith('_'):
            self.namespace[k] = v
        elif k.startswith('D_'):
            self.declare_derivative(k[2:], v)
        elif k == 'bounds':
            self.declare_bounds(v)
        else:
            self.declare_value(k, v)

    def _add_symbol(self, name):
        self.symbols.add(name)
        self.namespace[name] = Symbol(name, real=True)

    def declare_derivative(self, name, spec):
        """
        Derivative declarations automatically promotes value to a dynamic
        variable. Spec can be a constant, a function, or an expression.
        """
        if name not in self.values:
            raise TypeError(f'derivative of unknown variable: D_{name}')
        self.equations[name] = spec
        self._add_symbol('D_' + name)

    def declare_value(self, name, value):
        """
        Value declarations can have many different forms

        ``name = value``:
            ...
        ``name = value, '[unit] description``:
            ...
        """
        unit = DIMENSIONLESS
        msg = ''

        if isinstance(value, tuple):
            value, spec = value
            unit, msg = parse_unit_msg(spec)
        if not isinstance(value, Value):
            value = Value(name, value, unit=unit, description=msg)

        self.values[name] = value
        self._add_symbol(name)

    def declare_bounds(self, bounds):
        """
        Declare variable bounds.
        """

        # Lists are spliced
        if isinstance(bounds, (list, tuple, set)):
            for boundary in bounds:
                self.declare_bounds(boundary)
            return

        # Bound is expressed as an inequality such as x > y
        if isinstance(bounds, Relational):
            self._declare_relational_bound(bounds)
        else:
            raise NotImplementedError(bounds)

    def _declare_relational_bound(self, bound):
        lhs, rhs = bound.lhs, bound.rhs

        # Auxiliary functions
        gt = lambda x: isinstance(x, (sympy.StrictGreaterThan, sympy.GreaterThan))
        lt = lambda x: isinstance(x, (sympy.StrictLessThan, sympy.LessThan))

        # Normalize "number {op} expr" or "other {op} symbol"
        if is_numeric(lhs) or isinstance(rhs, Symbol):
            rhs, lhs = lhs, rhs

        # symbol {op} number
        if isinstance(lhs, Symbol) and is_numeric(rhs):
            if gt(bound):
                self.lower[str(lhs)] = rhs
            elif lt(bound):
                self.upper[str(lhs)] = rhs
            else:
                raise TypeError(f'invalid bound expression: {bound}')

        # Other bound
        else:
            raise NotImplementedError(bound)


    def finalize(self):
        """
        Return a namespace dictionary used to create the Model subclass.
        """
        symbols = self.symbols
        return {k: v for k, v in self.namespace.items() if k not in symbols}
