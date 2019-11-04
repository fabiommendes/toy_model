from numbers import Number

import numpy as np
from sympy import Symbol, Expr, lambdify, S
from typing import Mapping

from ..utils import is_numeric


class Compiler:
    """
    Compiler is responsible for creating functions to calculate the derivative
    and computed values.
    """

    def __init__(self, dynamic, computed, equations, dtype=np.float64):
        self.dtype = dtype
        self.vars = dynamic
        self.aux = computed
        self.equations = equations

        self._idx_vars = {k: i for i, k in enumerate(self.vars)}
        self._idx_aux = {k: i for i, k in enumerate(self.aux)}
        self._var_size = sum(v.size for v in self.vars.values())
        self._aux_size = sum(v.size for v in self.aux.values())

    def vectorize_vars(self, m: Mapping[str, Number]) -> np.ndarray:
        """
        Vectorize dictionary mapping from var names to values.
        """
        return self._vectorize(m, self._idx_vars, self._var_size)

    def vectorize_aux(self, m: Mapping[str, Number]) -> np.ndarray:
        """
        Vectorize dictionary mapping from aux names to values.
        """
        return self._vectorize(m, self._idx_aux, self._aux_size)

    def _vectorize(self, data, idx, n):
        res = np.empty(n, dtype=self.dtype)
        for k, v in data.items():
            res[idx[k]] = v
        return res

    def var_map(self):
        return self._idx_vars.copy()

    def aux_map(self, absolute=False):
        if absolute:
            return self._idx_aux.copy()
        else:
            s = self._var_size
            return {k: v + s for k, v in self._idx_aux.items()}

    def var_index(self, attr):
        return self._idx_vars[attr]

    def aux_index(self, attr, absolute=False):
        if absolute:
            return self._idx_aux[attr] + self._var_size
        else:
            return self._idx_aux[attr]

    def compile_update_diff_fn(self):
        """
        Return the derivative updater function calculates the computed terms
        from a state array.

        The updater function has the signature ``fn(diff, y, x, t) -> None``
        in which ``diff`` is the output array of the same size of state ``x``,
        ``y`` is the
        """
        idx = self._idx_vars
        functions = tuple((idx[k], self._get_diff_fn(k)) for k in self.vars)

        def update_diff(diff, y, x, t):
            for i, fn in functions:
                yi = fn(t, y, x)
                diff[i] = yi

        return update_diff

    def compile_update_aux_fn(self):
        """
        Return a function that computes the computed terms from a state array.
        """
        idx = self._idx_aux
        functions = tuple((idx[k], self._get_aux_fn(k)) for k in self.aux)

        def update_computed(y, x, t):
            try:
                for i, fn in functions:
                    yi = fn(t, y, x)
                    y[i] = yi
            except Exception as exc:
                name = type(exc).__name__
                msg = f'{name} error occurred when calling {fn.__name__!r}: {exc}'
                raise ValueError(msg) from exc

        return update_computed

    def compile_diff_fn(self, require_computed=False):
        """
        Create function that computes the derivative from state and time.

        If ``required_computed=True`` it will additionally take a vector with
        the value of computed values as an additional parameter.
        """
        update = self.compile_update_diff_fn()
        empty_vars = np.zeros(self._var_size, dtype=self.dtype).copy

        if require_computed:
            def diff(t, y, x):
                out = empty_vars()
                update(out, y, x, t)
                return out
        else:
            update_computed = self.compile_update_aux_fn()
            empty_computed = np.zeros(self._aux_size, dtype=self.dtype).copy

            def diff(t, x):
                y = empty_computed()
                update_computed(y, x, t)
                out = empty_vars()
                update(out, y, x, t)
                return out

        return diff

    def compile_aux_fn(self):
        update = self.compile_update_aux_fn()
        empty_computed = np.zeros(self._aux_size, dtype=self.dtype).copy

        def computed(t, x):
            y = empty_computed()
            update(y, x, t)
            return y

        return computed

    def _get_diff_fn(self, name):
        fn = self._get_fn(name, self.equations[name])
        fn.__name__ = fn.__qualname__ = f'eq/{name}'
        return fn

    def _get_aux_fn(self, name):
        fn = self._get_fn(name, self.aux[name].value)
        fn.__name__ = fn.__qualname__ = f'aux/{name}'
        return fn

    def _get_fn(self, name, expr):
        if is_numeric(expr):
            return self._get_numeric_fn(name, expr)
        elif isinstance(expr, Symbol):
            return self._get_symbol_fn(name, expr)
        elif isinstance(expr, Expr):
            return self._get_symbolic_expr_fn(name, expr)
        elif callable(expr):
            return self._get_callable_fn(name, expr)
        else:
            raise TypeError(f'invalid value for {name}: {expr}')

    def _get_numeric_fn(self, name, value):
        number = float(value)
        return lambda t, y, x: number

    def _get_symbol_fn(self, name, symb):
        if symb.name in self.vars:
            idx = self._idx_vars[symb.name]
            return lambda t, y, x: x[idx]
        elif symb.name in self.aux:
            idx = self._idx_aux[symb.name]
            return lambda t, y, x: y[idx]
        else:
            raise ValueError(f'invalid variable for {name}: {symb.name}')

    def _get_symbolic_expr_fn(self, name, expr):
        deps = set(map(str, filter(lambda x: isinstance(x, Symbol), expr.atoms())))
        args = (
            S('t'),
            *(k for k in self.vars if k in deps),
            *(k for k in self.aux if k in deps),
        )
        lambd = lambdify(args, expr)

        args_var = np.array([self._idx_vars[k] for k in self.vars if k in deps],
                            dtype=int)
        args_aux = np.array([self._idx_aux[k] for k in self.aux if k in deps], dtype=int)

        def fn(t, y, x):
            a = y[args_aux]
            b = x[args_var]
            return lambd(t, *a, *b)

        return fn

    def _get_callable_fn(self, name, expr):
        raise NotImplementedError(name, expr)
