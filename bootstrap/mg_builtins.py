import sys

import syntax

builtins = {}

def mg_builtin(name):
    def annotate(fn):
        builtins[name] = syntax.BuiltinFunction(name, fn)
        return fn
    return annotate

@mg_builtin('getchar')
def mgb_getchar(ctx, args):
    return syntax.String(sys.stdin.read(1))

@mg_builtin('putchar')
def mgb_putchar(ctx, args):
    sys.stdout.write(args[0].name)
    return syntax.Nil()

__all__ = builtins
