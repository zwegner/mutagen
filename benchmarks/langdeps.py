# Compatibility layer for differences in Python/Mutagen

# Pretty much a straight copy from Mutagen builtins. meh.
def fixed_point(fn):
    def recurse(x):
        return x(x)
    def partial(f, arg):
        def applied(*args):
            return f(arg, *args)
        return applied
    def call(rec):
        def inner(*args):
            return rec(rec)(*args)
        return partial(fn, inner)
    return recurse(call)
