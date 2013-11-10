# Test parse_int
for [input, base, result] in [['0xf', 0, 15], ['077', 8, 63], ['-123xyz', 36, -64009403]]:
    assert parse_int(input, base) == result

# Test str_upper
for [reg, upper] in [['Hello World!', 'HELLO WORLD!'], ['HELLO', 'HELLO']]:
    assert str_upper(reg) == upper

# Test range
assert list(range(0)) == []
assert list(range(4)) == [0, 1, 2, 3]

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
assert str_join('/', str_list) == str_joined
assert str_split(str_joined, '/') == str_list
