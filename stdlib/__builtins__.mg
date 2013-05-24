# HACK!!! This is to get around a lack of closures/lifting,
# by manually passing around the function.
def map(fn, list):
    def reducer(a, b):
        fn = a[0]
        [fn, a[1] + [fn(b)]]
    reduce(reducer, [fn, []], list)[1]

def print(args):
    putchar(str(args) + '\n')

def str_split(text, delim):
    c = 0
    r = []
    while c < len(text):
        start = c
        while c < len(text) and text[c] != delim:
            c = c + 1
        r = r + [slice(text, start, c)]
        c = c + 1
    r

def str_split_lines(text):
    str_split(text, '\n')

class set:
    def __init__(items):
        set_items = []
        for i in items:
            if i not in set_items:
                set_items = set_items + [i]
        make(['items', set_items])
    def add(self, item):
        set(self.items + [item])
    def __or__(self, other):
        set(self.items + other.items)
    def __repr__(self):
        'set('+repr(self.items)+')'
