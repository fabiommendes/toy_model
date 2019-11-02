import numpy as np


def namedarray(name, fields, extra=None):
    """
    Creates a new named-array class. It is analogous as a named tuple,
    but it names fields of a numpy array.

    Differently from regular arrays, this subclass has a fixed size.
    """

    fields = np.asarray(fields)
    shape = fields.shape
    index_map = {fields[idx]: idx for idx in iter_args(fields)}
    ns = {k: make_property(idx) for k, idx in index_map.items()}
    ns.update(extra or {})

    def __new__(cls, *args, dtype=None, copy=False, order=None, **kwargs):
        if args and kwargs:
            raise ValueError('cannot pass values as positional and keyword arguments.')
        elif kwargs:
            data = np.zeros(shape, dtype=dtype)
            for k, v in kwargs.items():
                idx = index_map[k]
                data[idx] = v
        elif copy:
            data, = args
            data = np.asanyarray(data).copy()
        else:
            data, = args
            data = np.asanyarray(data)

        if not data.shape[:len(shape)] == shape:
            raise ValueError(f'incompatible shapes: {data.shape} to {shape}')
        strides = data.__array_interface__['strides']
        new = np.ndarray.__new__(cls, data.shape, dtype=dtype or float, strides=strides,
                                 order=order)
        new[:] = data
        return new

    ns['__new__'] = __new__
    return type(name, (np.ndarray,), ns)


def make_property(idx):
    """
    Creates a property accessor for a named position in array.
    """

    def fget(self):
        return self[idx]

    def fset(self, value):
        self[idx] = value

    return property(fget, fset)


def iter_args(data, kind=(list, tuple, np.ndarray)):
    """
    Iterator of index tuples for multidimensional array data.
    """

    for i, v in enumerate(data):
        if isinstance(v, kind):
            for idx in iter_args(v, kind):
                yield (i, *idx)
        else:
            yield (i,)
