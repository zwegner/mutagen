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

def map(fn, *lists):
    for args in zip(*lists):
        yield fn(*args)

def reversed(iterable):
    i = len(iterable)
    while i > 0:
        i = i - 1
        yield iterable[i]

def range(end):
    i = 0
    while i < end:
        yield i
        i = i + 1

def enumerate(gen):
    i = 0
    for item in gen:
        yield [i, item]
        i = i + 1

def zip(*iterables):
    # HACK: the iterables are eagerly evaluated, so this takes up too much
    # memory/time.
    # HACK?: can't use map here, since it calls zip...
    iters = []
    for iterable in iterables:
        iters = iters + [list(iterable)]

    i = 0
    done = len(iters) == 0
    while not done:
        item = []
        for iterable in iters:
            if i >= len(iterable):
                done = True
                break
            item = item + [iterable[i]]
        if not done:
            yield item
            i = i + 1

def print(*args):
    return putstr(str_join(' ', map(str, args)) + '\n')

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

def str_join(sep, strs):
    r = ''
    first = True
    for s in strs:
        if first:
            first = False
        else:
            r = r + sep
        r = r + s
    return r

def str_starts_with(a, b):
    return slice(a, 0, len(b)) == b

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
