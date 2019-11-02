import numpy as np
from sympy import Symbol, Expr

import sidekick as sk
from ..utils import is_numeric


class Compiler:
    """
    Compiler is responsible for creating functions to calculate the derivative
    and computed values.
    """

    def __init__(self, dynamic, computed, equations, dtype=np.float64):
        self.dtype = dtype
        self.vars = dynamic
        self.computed_terms = computed
        self.equations = equations

        self._idx_vars = {k: i for i, k in enumerate(self.vars)}
        self._idx_computed = {k: i for i, k in enumerate(self.computed_terms)}
        self._n_vars = sum(v.size for v in self.vars.values())
        self._n_computed = sum(v.size for v in self.computed_terms.values())

    def initial_state(self, state):
        """

        """
        x0 = np.zeros(self._n_vars, dtype=self.dtype)
        idx = self._idx_vars
        for k, v in state.items():
            x0[idx[k]] = v
        return x0

    def compile_update_diff_fn(self):
        """
        Return the derivative updater function calculates the computed terms
        from a state array.

        The updater function has the signature ``fn(diff, y, x, t) -> None``
        in which ``diff`` is the output array of the same size of state ``x``,
        ``y`` is the
        """
        idx = self._idx_vars
        functions = tuple((idx[k], self._get_derivative_fn(k)) for k in self.vars)

        def update_diff(diff, y, x, t):
            for i, fn in functions:
                diff[i] = fn(y, x, t)

        return update_diff

    def compile_update_computed_terms_fn(self):
        """
        Return a function that computes the computed terms from a state array.
        """
        idx = self._idx_computed
        functions = tuple((idx[k], self._get_computed_fn(k)) for k in self.computed_terms)

        def update_computed(y, x, t):
            for i, fn in functions:
                y[i] = fn(y, x, t)

        return update_computed

    def compile_diff_fn(self, require_computed=False):
        """
        Create function that computes the derivative from state and time.

        If ``required_computed=True`` it will additionally take a vector with
        the value of computed values as an additional parameter.
        """
        update = self.compile_update_diff_fn()
        empty_vars = np.zeros(self._n_vars, dtype=self.dtype).copy

        if require_computed:
            def diff(y, x, t):
                out = empty_vars()
                update(out, y, x, t)
                return out
        else:
            update_computed = self.compile_update_computed_terms_fn()
            empty_computed = np.zeros(self._n_computed, dtype=self.dtype).copy

            def diff(x, t):
                y = empty_computed()
                update_computed(y, x, t)

                out = empty_vars()
                update(out, y, x, t)
                return out

        return diff

    def compile_computed_terms_fn(self):
        update = self.compile_update_computed_terms_fn()
        empty_computed = np.zeros(self._n_computed, dtype=self.dtype).copy

        def computed(x, t):
            y = empty_computed()
            update(y, x, t)
            return y

        return computed

    def _get_derivative_fn(self, name):
        return self._get_fn(name, self.equations[name])

    def _get_computed_fn(self, name):
        return self._get_fn(name, self.computed_terms[name])

    def _get_fn(self, name, expr):
        print('fn:', name, expr)

        if is_numeric(expr):
            return self._get_numeric_fn(name, expr)
        elif isinstance(expr, Symbol):
            return self._get_symbol_fn(name, expr)
        elif isinstance(expr, Expr):
            return self._get_symbolic_expr_fn(name, expr)
        elif callable(expr.value):
            return self._get_callable_fn(name, expr)
        else:
            raise TypeError('invalid value')

    def _get_numeric_fn(self, name, value):
        number = float(value)
        return lambda y, x, t: number

    def _get_symbol_fn(self, name, symb):
        if symb.name in self.vars:
            idx = self._idx_vars[symb.name]
            return lambda y, x, t: x[idx]
        elif symb.name in self.computed_terms:
            idx = self._idx_computed[symb.name]
            return lambda y, x, t: y[idx]
        else:
            raise ValueError(f'invalid variable for {name}: {symb.name}')

    def _get_symbolic_expr_fn(self, name, expr):
        deps = expr.atoms()
        deps = expr.dependent_variables()
        state, computed = map(tuple, sk.separate(self.vars.__contains__, deps))
        raise NotImplementedError(expr)

    def _get_callable_fn(self, name, expr):
        raise NotImplementedError(name, expr)