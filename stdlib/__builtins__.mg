################################################################################
## Mutagen builtin functions ###################################################
################################################################################

def reduce(fn, start, iter):
    for i in iter:
        start = fn(start, i)
    return start

def foldl(fn, list, nil):
    for i in list:
        nil = fn(i, nil)
    return nil

def foldr(fn, list, nil):
    for i in reversed(list):
        nil = fn(i, nil)
    return nil

def map(fn, list):
    result = []
    for i in list:
        result = result + [fn(i)]
    return result

def reversed(iterable):
    i = len(iterable)
    while i > 0:
        i = i - 1
        yield iterable[i]

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

# XXX need a real class
def list(items):
    l = []
    for i in items:
        l = l + [i]
    return l

def type(obj):
    return obj.__class__

def isinstance(obj, cls):
    return type(obj) == cls

################################################################################
## Python-implemented builtins #################################################
################################################################################

def read_file(path):
    return py_obj_get('open')(path).read()

def putchar(arg):
    return py_obj_get('sys.stdout.write')(arg)

def str_upper(arg):
    return py_obj_get('str.upper')(arg)

def parse_int(int_str, base):
    return py_obj_get('int')(int_str, base)

################################################################################
## Builtin classes #############################################################
################################################################################

class set:
    def __init__(items):
        items = map(py_wrap, items)
        return {'items': py_obj_get('set')(items)}
    def add(self, item):
        return self | set([item])
    def __or__(self, other):
        return set(self.items.__or__(other.items))
    def __repr__(self):
        s = '{'
        first = True
        for i in self:
            if not first:
                s = s + ', '
            s = s + repr(i)
            first = False
        return s + '}'
    def __iter__(self):
        for i in self.items.__iter__():
            yield i
    def __contains__(self, item):
        return self.items.__contains__(py_wrap(item))
    def __eq__(self, other):
        return self.items == other.items
    def __len__(self):
        return self.items.__len__()
    def __bool__(self):
        return len(self) > 0
