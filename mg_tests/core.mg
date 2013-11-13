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

# Test union
union TestUnion(x, y):
    pass
x = TestUnion.x()
y = TestUnion.y()
assert isinstance(x, TestUnion.x)
assert not isinstance(x, TestUnion.y)
assert not isinstance(y, TestUnion.x)
assert isinstance(y, TestUnion.y)
# Need better isinstance
#assert isinstance(x, TestUnion)
