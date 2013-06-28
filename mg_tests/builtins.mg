# Test parse_int
for i in [['0xf', 0, 15], ['077', 8, 63], ['-123xyz', 36, -64009403]]:
    assert parse_int(i[0], i[1]) == i[2]

# Test str_upper
for i in [['Hello World!', 'HELLO WORLD!'], ['HELLO', 'HELLO']]:
    assert str_upper(i[0]) == i[1]

# Test enumerate
assert list(enumerate('abc')) == [[0, 'a'], [1, 'b'], [2, 'c']]

# Test set
items_1 = [0, 1, 2, 3, 4, {'a': 'b'}]
items_2 = [5, 6, 7, 8, 9, {'c': 'd'}]
s = set(items_1)
t = set(items_2)
u = s | t
assert u | s == u
assert u | t == u
assert u and s and t
assert not set([])

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
