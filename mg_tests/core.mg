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
    # XXX this is not working now, for a rather complicated reason related to
    # lambda lifting. Since the test function in the loop has a variable from
    # the parent scope, it gets lifted out to the beginning of the program. At
    # that point, type is not declared yet. This can be sort-of solved by making
    # sure to insert the lifted lambda in the right place, in the top-most
    # statement list just before the current expression tree. However, this
    # exposes a deeper problem: the style of lambda lifting we use needs two
    # different kinds of scope-borrowing: one for the function definition, which
    # includes parameter types, the return type, and eventually any expression
    # used as a default argument (once that's supported), and one for the actual
    # body of the function. Since lifted lambdas use the same BoundFunction type
    # we use for bound-methods to pass in arguments to the function, this does
    # not apply to the actual function. As it is, we create only one Function
    # object for whole function, with specializations coming from different
    # BoundFunction instantiations. But since the parameter/return types and
    # default arguments can change too, we really need full new Function
    # instantiations, or at least some other wrapper that does the type checking
    # and argument handling.
    #for y in test_types:
    #    def test_ret() -> type(x):
    #        return y
    #    if isinstance(y, type(x)):
    #        assert isinstance(test_ret(), type(x))
    #        assert_call_fails(assert_call_fails, test_ret)
    #    else:
    #        assert_call_fails(test_ret)

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
assert str_starts_with(str(t), '<TestClass at ')

assert isinstance(type(t), type)
assert isinstance(type(type(t)), type)

# Test union
union TestUnion(x, y):
    pass
x = TestUnion.x()
y = TestUnion.y()
assert isinstance(x, TestUnion.x)
assert not isinstance(x, TestUnion.y)
assert isinstance(y, TestUnion.y)
assert not isinstance(y, TestUnion.x)
assert str_starts_with(str(x), '<TestUnion.x at ')
assert str_starts_with(str(y), '<TestUnion.y at ')
# Need better isinstance
#assert isinstance(x, TestUnion)

# Test union with types and inline class definition
# XXX change type of a to TestUnion when there's a better isinstance
union TestNestedUnion(a: TestUnion.x, b: class(x, y)):
    pass
a = TestNestedUnion.a(x)
b = TestNestedUnion.b(x, y)
assert isinstance(a, TestNestedUnion.a)
assert not isinstance(a, TestNestedUnion.b)
assert isinstance(b, TestNestedUnion.b)
assert isinstance(b.x, TestUnion.x)
assert isinstance(b.y, TestUnion.y)
assert not isinstance(b, TestNestedUnion.a)
assert str_starts_with(str(a), '<TestNestedUnion.a at ')
assert str_starts_with(str(b), '<TestNestedUnion.b at ')
assert {TestNestedUnion.b(x, y): TestNestedUnion.a(x)}[b] == a

union Bool(false, true):
    def __not__(self):
        return {Bool.false(): Bool.true(), Bool.true(): Bool.false()}[self]
assert Bool.false().__not__() == Bool.true()
assert Bool.false().__not__().__not__() == Bool.false()
assert Bool.true().__not__() != Bool.true()

union Maybe(Nothing, Just: class(value)):
    pass
nothing = Maybe.Nothing()
just = Maybe.Just(7)
assert just.value == 7
