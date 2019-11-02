from numbers import Number

from sidekick import lazy, placeholder as _
from sympy import Symbol


class Meta:
    state_size = lazy(_.storage_size + _.params_size)
    storage_size = lazy(_.dynamic_variables_size + _.computed_size)
    values = lazy(lambda _: {**_.dynamic_variables, **_.computed, **_.params})

    def __init__(self, model):
        self.model = model
        #self._init_value_categories()


