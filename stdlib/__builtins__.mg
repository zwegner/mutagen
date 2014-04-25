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

def range(*args):
    [start, step] = [0, 1]
    if len(args) == 1:
        [end] = args
    elif len(args) == 2:
        [start, end] = args
    elif len(args) == 3:
        [start, end, step] = args
    else:
        error('bad arguments to range()')
    if step == 0:
        error('step argument to range() must be nonzero')
    i = start
    while (step > 0 and i < end) or (step < 0 and i > end):
        yield i
        i = i + step

def enumerate(gen):
    i = 0
    for item in gen:
        yield [i, item]
        i = i + 1

def any(gen):
    for item in gen:
        if item:
            return True
    return False

def all(gen):
    for item in gen:
        if not item:
            return False
    return True

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
    return putstr(' '.join(map(str, args)) + '\n')

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

def str_starts_with(a, b):
    return slice(a, 0, len(b)) == b

def isinstance(obj, cls):
    return type(obj) == cls
