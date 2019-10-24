from toy.namedarray import namedarray, iter_args


class TestNamedArray:
    def test_iter_args(self):
        assert list(iter_args([3, 2, 1])) == [(0,), (1,), (2,)]
        assert list(iter_args([[3, 2], [1, 2]])) == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_named_array_creation(self):
        Point = namedarray('Point', ['x', 'y'])
        pt = Point([1, 2])

        # Check read
        assert pt.x == 1
        assert pt.y == 2

        # Check write
        pt.y = 3
        assert (pt == [1, 3]).all()