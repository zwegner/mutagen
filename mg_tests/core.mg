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
