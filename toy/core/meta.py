from sidekick import lazy, delegate_to
from ..compiler import Compiler


class Meta:
    """
    Holds meta-information about model.
    """

    # Delegated
    vars = delegate_to('model')
    aux = delegate_to('model')
    params = delegate_to('model')
    compile_diff_fn = delegate_to('compiler')
    compile_aux_fn = delegate_to('compiler')

    # Computed variables
    diff_fn = lazy(lambda self: self.compile_diff_fn())
    aux_fn = lazy(lambda self: self.compile_aux_fn())
    vars_size = lazy(lambda self: sum(v.size for v in self.vars.values()))
    aux_size = lazy(lambda self: sum(v.size for v in self.aux.values()))
    params_size = lazy(lambda self: sum(v.size for v in self.params.values()))
    t0 = 0.0
    tf = 10.0
    steps = 100
    y0 = property(lambda self: self.compiler.vectorize_vars(self.model.var_values()))

    @lazy
    def compiler(self):
        m = self.model
        return Compiler(m.vars, m.aux, m.equations, dtype=m.dtype)

    def __init__(self, model):
        self.model = model

    def unvectorize_vars(self, y):
        """
        Convert vector state to a dictionary.
        """
        map = self.compiler.var_map()
        return {name: y[idx] for name, idx in map.items()}

    def read_var(self, name, src):
        """
        Read named var from source array.
        """
        idx = self.compiler.var_index(name)
        return src[idx]

    def read_aux(self, name, src):
        """
        Read named auxiliary term from source array.
        """
        idx = self.compiler.aux_index(name)
        return src[idx]
