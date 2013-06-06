for i in [['0xf', 0, 15], ['077', 8, 63], ['-123xyz', 36, -64009403]]:
    assert parse_int(i[0], i[1]) == i[2]

for i in [['Hello World!', 'HELLO WORLD!'], ['HELLO', 'HELLO']]:
    assert str_upper(i[0]) == i[1]

assert list(enumerate('abc')) == [[0, 'a'], [1, 'b'], [2, 'c']]

items_1 = [0, 1, 2, 3, 4]
items_2 = [5, 6, 7, 8, 9]
s = set(items_1)
t = set(items_2)
u = s | t
assert len(u) == 10

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
