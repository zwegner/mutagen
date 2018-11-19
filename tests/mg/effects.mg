class Yield():
    pass

class Yield2():
    pass

values = []
x = 1
consume:
    for i in range(2):
        consume:
            for i in range(2):
                values = values + [(perform Yield())]
            for i in range(2):
                values = values + [(perform Yield2())]
        effect Yield2 as y:
            x = x + 1
            resume x
effect Yield as y:
    x = x * 2
    resume x

assert values == [2, 4, 5, 6, 12, 24, 25, 26]
