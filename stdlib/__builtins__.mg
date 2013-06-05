def reduce(fn, start, iter):
    for i in iter:
        start = fn(start, i)
    return start

def map(fn, list):
    # HACK!!! This is to get around a lack of closures/lifting,
    # by manually passing around the function.
    def reducer(a, b):
        fn = a[0]
        return [fn, a[1] + [fn(b)]]
    return reduce(reducer, [fn, []], list)[1]

def enumerate(gen):
    i = 0
    for item in gen:
        yield [i, item]
        i = i + 1

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

def isinstance(obj, cls):
    return obj.__class__ == cls

class set:
    def __init__(items):
        set_items = []
        for i in items:
            set_items = set_items + [i]
        return {'items': py_obj_call(py_obj_get('set'), set_items)}
    def add(self, item):
        return self | set([item])
    def __or__(self, other):
        return set(py_obj_call(self.items.__or__, other.items))
    def __repr__(self):
        return py_obj_call(self.items.__repr__)
    def __iter__(self):
        for i in self.items.__iter__():
            yield i
    def __bool__(self):
        return py_obj_call(self.items.__len__) > 0
