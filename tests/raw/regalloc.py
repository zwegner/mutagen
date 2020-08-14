import sys

from bootstrap import regalloc

def test_parallel_copy():
    def test(read, write, expected):
        ops = regalloc.parallel_copy(read, write, set())
        assert ops == expected, ('regalloc test failure:\n     got: %s\n'
                'expected: %s' % (ops, expected))

    test([1, 1, 2, 5, 4], [1, 2, 3, 4, 5],
            [('mov', 3, 2), ('mov', 2, 1), ('mov', 1, 5), ('mov', 5, 4),
            ('mov', 4, 1)])

    test([1, 1, 2, 6, 4, 5], [1, 2, 3, 4, 5, 6],
            [('mov', 3, 2), ('mov', 2, 1), ('mov', 1, 6), ('mov', 6, 5),
            ('mov', 5, 4), ('mov', 4, 1)])

    test([2, 1], [1, 2],
            [('swap', 2, 1)])

    test([2, 3, 1], [1, 2, 3],
            [('swap', 3, 1), ('swap', 2, 1)])

    test([1, 2, 3], [2, 3, 1],
            [('swap', 2, 1), ('swap', 3, 1)])

    test([1, 2, 3, 4], [2, 3, 4, 1],
            [('swap', 2, 1), ('swap', 3, 1), ('swap', 4, 1)])

test_parallel_copy()
