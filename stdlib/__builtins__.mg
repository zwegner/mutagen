def map(fn, list):
    # HACK!!! This is to get around a lack of closures/lifting,
    # by manually passing around the function.
    def reducer(a, b):
        fn = a[0]
        return [fn, a[1] + [fn(b)]]
    return reduce(reducer, [fn, []], list)[1]

def list(arg):
    l = []
    for a in arg:
        l = l + [a]
    return l

def print(args):
    return putchar(str(args) + '\n')

def str_split(text, delim):
    c = 0
    r = []
    while c < len(text):
        start = c
        while c < len(text) and text[c] != delim:
            c = c + 1
        r = r + [slice(text, start, c)]
        c = c + 1
    return r

def str_split_lines(text):
    return str_split(text, '\n')

class set:
    def __init__(items):
        set_items = []
        for i in items:
            if i not in set_items:
                set_items = set_items + [i]
        return make(['items', set_items])
    def add(self, item):
        return set(self.items + [item])
    def __or__(self, other):
        return set(self.items + other.items)
    def __repr__(self):
        return 'set('+repr(self.items)+')'
