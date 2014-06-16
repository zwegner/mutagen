# Test conditional expressions
def test_cond_expr(cond, x, y):
    return x if cond else y
assert test_cond_expr(True, 1, 2) == 1
assert test_cond_expr(False, 1, 2) == 2
# Make sure they're parsed properly
assert 0 and 0 if False else 1 or 0 if True else 0 and 1

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
def test_fn(a: int, b: str):
    return str(a) + b
assert_call_fails(lambda { test_fn('a', 'b'); })
assert_call_fails(lambda {
    assert_call_fails(lambda { test_fn(1, 'b'); }); })

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

# Test fat-arrow lambdas
test_lambda = (x, y, exp=2) => x ** exp + y ** exp
assert test_lambda(3, 4) == 25
assert test_lambda(3, 4, exp=3) == 91

# Test lambda lifting--static closures mean variables capture value at time of
# definition
items = list(range(4))
lambdas = []
for x in items:
    lambdas = lambdas + [lambda { return x; }]

for [v, l] in zip(items, lambdas):
    assert v == l()

# Slightly more complicated case, make sure multiple bound variables and
# also parameters work properly
lambdas = []
for x in items:
    for y in items:
        lambdas = lambdas + [lambda (z) { assert z == x * len(items) + y; }]

for [v, l] in zip(range(16), lambdas):
    l(v)

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
