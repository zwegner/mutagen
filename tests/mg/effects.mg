class Eff1():
    pass

class Eff2():
    pass

# Test 1: test multiple effect handlers along with interaction with other control flow
def control_flow_1(values):
    consume:
        values = values + [(perform Eff1())]
        values = values + [(perform Eff2())]
        return values
    effect Eff1 as y:
        resume 1

values = []
for i in range(3):
    consume:
        values = control_flow_1(values)
        values = control_flow_1(values)
        if i == 1:
            break
    effect Eff2 as y:
        resume 2

assert values == [1, 2, 1, 2, 1, 2, 1, 2]

# Test 2: test interaction with effects and generators (which are handled by effects internally)
def gen_1():
    for i in range(4):
        yield (perform Eff1())
        yield (perform Eff2())
        yield i

def gen_2():
    consume:
        for i in gen_1():
            yield i
    effect Eff2 as y:
        resume -2

def gen_3():
    consume:
        for i in gen_2():
            yield i
    effect Eff1 as y:
        resume -1

values = list(gen_3())
assert values == [-1, -2, 0, -1, -2, 1, -1, -2, 2, -1, -2, 3]

# Test 3: test state handling in the presence of effects
values = []
x = 1
consume:
    for i in range(2):
        consume:
            for i in range(2):
                values = values + [(perform Eff1())]
            for i in range(2):
                values = values + [(perform Eff2())]
        effect Eff2 as y:
            x = x + 1
            resume x
effect Eff1 as y:
    x = x * 2
    resume x

expected = [2, 4, 5, 6, 12, 24, 25, 26]
assert values == expected

# Test 4: like test 3, but with more nesting + generators
def g():
    def f():
        for i in range(2):
            yield i
    return f

def big_crazy():
    x = 1
    consume:
        for i in range(2):
            consume:
                for i in g()():
                    yield (perform Eff1())
                for i in g()():
                    yield (perform Eff2())
            effect Eff2 as y:
                x = x + 1
                resume x
    effect Eff1 as y:
        x = x * 2
        resume x

values = list(big_crazy())
assert values == expected

# Test 5: test unhandled effects
assert_call_fails(lambda(): (perform Eff1()))
