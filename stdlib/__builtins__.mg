################################################################################
## Mutagen builtin functions ###################################################
################################################################################

def reversed(iterable):
    i = len(iterable)
    while i > 0:
        i = i - 1
        yield iterable[i]

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

def filter(fn, list):
    for i in list:
        if fn(i):
            yield i

def max(iterable):
    highest = None
    for item in iterable:
        if highest == None or item > highest:
            highest = item
    return highest

def min(iterable):
    lowest = None
    for item in iterable:
        if lowest == None or item < lowest:
            lowest = item
    return lowest

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

def print(*args):
    return putstr(' '.join(map(str, args)) + '\n')

def isinstance(obj, cls):
    return type(obj) == cls

def sum(iterable, base):
    for item in iterable:
        base = base + item
    return base

def partial(fn, *args1):
    def applied(*args2):
        return fn(*args1, *args2)
    return applied

# XXX would be nice to use macro magic to add the name of the function as an
# implicit first parameter, so we wouldn't have to have an explicit alias name
def fixed_point(fn):
    def recurse(x):
        return x(x)
    def call(c):
        def inner(*args):
            return c(c)(*args)
        return partial(fn, inner)
    return recurse(call)

# Super fucking slow selection sort!
def sorted(iterable):
    iterable = list(iterable)
    for i in range(len(iterable)):
        best = 0
        for j in range(1, len(iterable)):
            if iterable[j] < iterable[best]:
                best = j
        yield iterable[best]
        iterable = slice(iterable, best) + slice(iterable, best + 1, None)

# This shit's slow
# XXX recursion
class set:
    def __init__(*items):
        if len(items) == 0:
            items = []
        else:
            [items] = items
        set_items = []
        for i in items:
            if i not in set_items:
                set_items = set_items + [i]
        return {'items': set_items}
    def __contains__(self, item):
        return item in self.items
    def __or__(self, other):
        return type(self)(self.items + other.items)
    def __sub__(self, other):
        new_items = []
        for item in self.items:
            if item not in other:
                new_items = new_items + [item]
        return type(self)(new_items)
    def pop(self):
        # Return the set without one item, and that item
        return [type(self)(slice(self.items, 1, None)), self.items[0]]
    def __iter__(self):
        for item in self.items:
            yield item
    def __len__(self):
        return len(self.items)
    def __eq__(self, other):
        return (isinstance(other, type(self)) and len(self) == len(other) and
            all([i in self.items for i in other.items]))
    def __repr__(self):
        if not self.items:
            return 'set()'
        return '{{{}}}'.format(', '.join(map(str, self.items)))
