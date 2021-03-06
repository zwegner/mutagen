# Test int
for [input, base, result] in [['0xf', 0, 15], ['077', 8, 63], ['-123xyz', 36, -64009403]]:
    assert int(input, base) == result
    assert isinstance(int(input, base), int)

# Test str()
for [input, result] in [[123, '123'], ['abc', 'abc'], [[0], '[0]']]:
    assert str(input) == result
    assert isinstance(str(input), str)

# Test str.upper
for [reg, upper] in [['Hello World!', 'HELLO WORLD!'], ['HELLO', 'HELLO']]:
    assert reg.upper() == upper

# Test bool
for [input, result] in [[False, False], [0, False], [9, True], ['0', True], ['', False],
        [[], False], [{'a': 1}, True]]:
    assert bool(input) == result

# Test range
assert list(range(-4)) == []
assert list(range(0)) == []
assert list(range(4)) == [0, 1, 2, 3]
assert list(range(-4, -2)) == [-4, -3]
assert list(range(-2, 2)) == [-2, -1, 0, 1]
assert list(range(5, 8)) == [5, 6, 7]
assert list(range(0, -5, -2)) == [0, -2, -4]
assert list(range(3, 6, 1)) == [3, 4, 5]
assert list(range(-4, 4, 3)) == [-4, -1, 2]
assert_call_fails(list, range(0, 0, 0))

# Test enumerate
assert list(enumerate('abc')) == [[0, 'a'], [1, 'b'], [2, 'c']]

test_list = [1, 2, 3, 4, 5]

# Test zip
assert list(zip()) == []
assert list(zip(test_list)) == list(map(lambda(x): [x], test_list))
assert list(zip(test_list)) != test_list
assert list(zip(test_list, test_list)) == [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
assert list(zip('abc', 'def')) == [['a', 'd'], ['b', 'e'], ['c', 'f']]

# Test reversed
assert list(reversed(test_list)) == [5, 4, 3, 2, 1]

# Test foldl/foldr
def join(a, b):
    return [a, b]

assert foldl(join, test_list, []) == [5, [4, [3, [2, [1, []]]]]]
assert foldr(join, test_list, []) == [1, [2, [3, [4, [5, []]]]]]

# Test slicing
assert test_list[2:] == [3, 4, 5]
assert test_list[:3] == [1, 2, 3]
assert test_list[2:4] == [3, 4]
assert test_list[::2] == [1, 3, 5]
assert test_list[-2::-1] == [4, 3, 2, 1]
assert test_list[:4:2] == [1, 3]
assert test_list[1:4:2] == [2, 4]

test_string = 'abcdef'
assert test_string[:3] == 'abc'
assert test_string[3:] == 'def'
assert test_string[2:-1] == 'cde'
assert test_string[::-1] == 'fedcba'
assert test_string[1::2] == 'bdf'
assert test_string[:3:-1] == 'fe'
assert test_string[1:4:2] == 'bd'

# Test str join/split
str_list = ['abc', 'def', 'ghi']
str_joined = 'abc/def/ghi'
assert '/'.join(str_list) == str_joined
assert str_joined.split('/') == str_list

# Test str.startswith
strs = ['', 'a', 'ab', 'abc']
other = '!a'
for [i, x] in enumerate(strs):
    for [j, y] in enumerate(strs):
        if j <= i:
            assert x.startswith(y)
        else:
            assert not x.startswith(y)
        assert not x.startswith(other)

assert not any([])
assert not any(range(1))
assert any(range(2))
assert not any([None])
assert all([])
assert not all(range(1))
assert not all(range(2))
assert all([1, True, [0]])

# Test the fixed-point combinator
@fixed_point
def fib(f, x):
    if x < 2:
        return 1
    return f(x - 2) + f(x - 1)

assert list(map(fib, range(10))) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

# Test sorted
test_sorted_list = [8, 1, -7, 3, 5, 9, 2, -10, -2, 0]
assert list(sorted(test_sorted_list)) == [-10, -7, -2, 0, 1, 2, 3, 5, 8, 9]
assert list(sorted(test_sorted_list, key=abs)) == [
    0, 1, 2, -2, 3, 5, -7, 8, 9, -10]
assert list(sorted(test_sorted_list, key=lambda(x): -x // 4)) == [
    9, 8, 5, 1, 3, 2, -2, 0, -7, -10]

# Test set
items_1 = [0, 1, 2, 3, 4, {'a': 'b'}]
items_2 = [5, 6, 7, 8, 9, {'c': 'd'}]
s = set(items_1)
t = set(items_2)
u = s | t
assert u | s == u
assert u | t == u
assert t | s == u
assert u and s and t
assert not set()

for i in items_1:
    assert i in s
    assert i not in t
    assert i in u
for i in items_2:
    assert i not in s
    assert i in t
    assert i in u
for i in u:
    assert i in s or i in t

for i in [-1, 'asdf', 99]:
    assert i not in u

assert len(s) == len(items_1)
assert len(t) == len(items_2)
assert len(u) == 2 * len(items_2)

# Test hacky inheritance
class A: r = 0
@hacky_inherit(A)
class B: r = 1; t = 0
@hacky_inherit(B)
class C: t = 1

a = A(); b = B(); c = C()
assert     isinstance(a, A) and     isinstance(b, A) and isinstance(c, A)
assert not isinstance(a, B) and     isinstance(b, B) and isinstance(c, B)
assert not isinstance(a, C) and not isinstance(b, C) and isinstance(c, C)
assert A.r == 0 and B.r == 1 and C.r == 1
assert B.t == 0 and C.t == 1
assert_call_fails(lambda: A.t)
