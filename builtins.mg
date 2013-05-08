# HACK!!! This is to get around a lack of closures/lifting,
# by manually passing around the function.
def map(fn, list):
    def reducer(a, b):
        fn = a[0]
        [fn, a[1] + [fn(b)]]
    reduce(reducer, [fn, []], list)[1]

def print(args):
    putchar(repr(args) + '\n')
