# Test conditional expressions
def test_cond_expr(cond, x, y):
    return x if cond else y
assert test_cond_expr(True, 1, 2) == 1
assert test_cond_expr(False, 1, 2) == 2
# Make sure they're parsed properly
assert 0 and 0 if False else 1 or 0 if True else 0 and 1

# Test dictionary addition/subtraction
test_dict = {'a': 1}
assert test_dict + {'b': 2} == {'a': 1, 'b': 2}
assert test_dict + {'a': -1} == {'a': -1}
assert test_dict - ['a'] == {}
assert_call_fails(lambda: test_dict - ['b'])

# Test dictionary iteration. This test needs all keys/values to have
# commutative operations, so just use ints. Note that dictionary iteration
# is different from Python so we have to compare against hardcoded results.
test_dict = {-1: 9, -3: 17, -5: 49}
[key_sum, value_sum] = [0, 0]
for [k, v] in test_dict:
    key_sum = key_sum + k
    value_sum = value_sum + v
assert key_sum == -9
assert value_sum == 75
assert key_sum == sum(test_dict.keys(), 0)
assert value_sum == sum(test_dict.values(), 0)

test_list = test_list_2 = [0, [0, [0, [0]]]]
test_list = test_list <- [1][1][1][0] = 1, [0] = 1
assert test_list == [1, [0, [0, [1]]]]
assert test_list_2 == [0, [0, [0, [0]]]]
assert 1 in test_list_2 <- [0] = 1

test_dict = {'a': 1, 'b': 2, 'c': 3}
test_dict = test_dict <- ['d'] = 4, ['e'] = 5
assert test_dict == {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}

test_set = {0, 0, 1, 1, 2, 2}
assert test_set == {0, 1, 2}
assert test_set == {2, 1, 0}
for i in range(3):
    [popped, new_set] = test_set.pop()
    assert popped in test_set
    assert popped not in new_set
    assert len(new_set) == len(test_set) - 1
    test_set = new_set
assert_call_fails(test_set.pop)

class TestClass(a, b):
    pass
test_obj = TestClass(0, [TestClass(0, 0)])
test_obj = test_obj <- .a = 1, .b[0].b = 2
assert test_obj == TestClass(1, [TestClass(0, 2)])
assert_call_fails(def() { test_obj <- .c = 0; })
test_obj = test_obj <- .b[0].a += 3, .b += [TestClass(1, -1)]
assert test_obj == TestClass(1, [TestClass(3, 2), TestClass(1, -1)])
assert_call_fails(def() { test_obj <- .b[0] += 0; })
assert_call_fails(def() { test_obj <- .b[0].b += 'a'; })

# Test that scoping works properly for assignments and for loops, that is,
# both of them should put their targets in the set of locals.
def test_scoping():
    x = [[0, 1]]
    for y in x:
        y
    for [y, z] in x:
        return y + z
test_scoping()

# Test parameter errors
def test_fn(a: int, b: str, c: int = 1234):
    return str(a) + b + str(c)
assert test_fn(0, 'b') == '0b1234'
assert test_fn(1, 'b', c=5) == '1b5'
assert_call_fails(lambda: test_fn('a', 'b'))
assert_call_fails(lambda: test_fn(0, 'b', c='c'))

# Test weird bug in parameter parsing: False as a default should still work
def test_fn(a, b, c=False):
    pass
test_fn(0, 1)

# Test return types
test_types = ['abc', 123, True, False]
for x in test_types:
    for y in test_types:
        def test_ret() -> type(x):
            return y
        if isinstance(y, type(x)):
            assert isinstance(test_ret(), type(x))
            assert_call_fails(assert_call_fails, test_ret)
        else:
            assert_call_fails(test_ret)

    def test_ret_bad() -> x:
        return 0
    assert_call_fails(test_ret_bad)

# Test keyword arguments. Different cases mainly since they're parsed separately
def kwargs0(x=4):
    return x
def kwargs1(base, x=4):
    return base * x
def kwargs2(*items, x=3):
    return sum(items, 0) * x
def kwargs3(base, *items, x=3):
    return sum(items, 0) * x + base
def kwargs4(init, **kwparams):
    return sum(kwparams.values(), init)
def kwargs5(x=3, **kwparams):
    assert 'x' not in kwparams
    return sum(sorted(kwparams.keys()), '') + str(x)
assert kwargs0() == 4
assert kwargs0(x=0) == 0
assert kwargs1(2) == 8
assert kwargs1(2, x=1) == 2
assert kwargs2(0, 1, 2) == 9
assert kwargs2(0, 1, 2, x=1) == 3
assert kwargs3(5, 0, 1, 2) == 14
assert kwargs3(5, 0, 1, 2, x=0) == 5
assert kwargs4(3, x=5, y=6, z=2) == 16
assert kwargs5(a=0, b=1, c=2, x=3) == 'abc3'

kw_dict = {'a': 1, 'b': 2, 'c': 3}
def pass_kwparams(**x): return x
assert pass_kwparams(**kw_dict) == kw_dict

# Test lambdas
test_lambda = lambda(x, y, exp=2): x ** exp + y ** exp
assert test_lambda(3, 4) == 25
assert test_lambda(3, 4, exp=3) == 91

# Test lambda lifting--static closures mean variables capture value at time of
# definition
items = list(range(4))
lambdas = []
for x in items:
    lambdas = lambdas + [lambda: x]

for [v, l] in zip(items, lambdas):
    assert v == l()

# Slightly more complicated case, make sure multiple bound variables and
# also parameters work properly
lambdas = []
for x in items:
    for y in items:
        lambdas = lambdas + [def(z) { assert z == x * len(items) + y; }]

for [v, l] in zip(range(16), lambdas):
    l(v)

# Test list comprehensions
expected = [[0, 1], [1, 0], [3, 0], [4, 1], [5, 0], [6, 1], [7, 0]]
v = [[x, y] for x in range(8) if x != 2 for y in range(2) if (x ^ y) & 1]
assert v == expected
# Same thing, but a dictionary comprehension
v = {x: y for x in range(8) if x != 2 for y in range(2) if (x ^ y) & 1}
assert list(v) == expected

# Test class
class TestA: pass
class TestB: pass
class TestClass(x: TestA, y: TestB, z=0):
    pass
t = TestClass(TestA(), TestB())
assert isinstance(t.x, TestA)
assert not isinstance(t.x, TestB)
assert not isinstance(t.y, TestA)
assert isinstance(t.y, TestB)
assert isinstance(t.z, int)
assert str(t).startswith('<TestClass at ')

assert isinstance(type(t), type)
assert isinstance(type(type(t)), type)
