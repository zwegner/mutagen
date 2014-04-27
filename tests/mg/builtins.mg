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
assert list(zip(test_list)) == list(map(lambda(x){return[x];}, test_list))
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

# Test str join/split
str_list = ['abc', 'def', 'ghi']
str_joined = 'abc/def/ghi'
assert '/'.join(str_list) == str_joined
assert str_split(str_joined, '/') == str_list

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
