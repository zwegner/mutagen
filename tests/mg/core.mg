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
assert_call_fails(lambda() { test_fn('a', 'b'); })
assert_call_fails(lambda() {
        assert_call_fails(lambda() { test_fn(1, 'b'); });
    })

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

# Test lambda lifting--static closures mean variables capture value at time of
# definition
items = list(range(4))
lambdas = []
for x in items:
    lambdas = lambdas + [lambda: return x;]

for [v, l] in zip(items, lambdas):
    assert v == l()

# Slightly more complicated case, make sure multiple bound variables and
# also parameters work properly
lambdas = []
for x in items:
    for y in items:
        lambdas = lambdas + [lambda (z): assert z == x * len(items) + y;]

for [v, l] in zip(range(16), lambdas):
    l(v)

# Test class
class TestA: pass
class TestB: pass
class TestClass(x: TestA, y: TestB):
    pass
t = TestClass(TestA(), TestB())
assert isinstance(t.x, TestA)
assert not isinstance(t.x, TestB)
assert not isinstance(t.y, TestA)
assert isinstance(t.y, TestB)
assert str(t).startswith('<TestClass at ')

assert isinstance(type(t), type)
assert isinstance(type(type(t)), type)
