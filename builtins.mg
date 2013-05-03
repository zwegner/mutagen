def map(fn, list):
    def reducer(a, b):
        a + [fn(b)]
    reduce(reducer, [], list)

def print(args):
    putchar(repr(args) + '\n')
