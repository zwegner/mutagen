import sys

import syntax

builtins = {}

def mg_builtin(name):
    def annotate(fn):
        builtins[name] = syntax.BuiltinFunction(name, fn)
        return fn
    return annotate

# TODO: error checking!

@mg_builtin('getchar')
def mgb_getchar(ctx, args):
    return syntax.String(sys.stdin.read(1))

@mg_builtin('putchar')
def mgb_putchar(ctx, args):
    sys.stdout.write(args[0].name)
    return syntax.Nil()

@mg_builtin('print')
def mgb_print(ctx, args):
    sys.stdout.write(args[0].name)
    return syntax.Nil()

@mg_builtin('len')
def mgb_len(ctx, args):
    return syntax.Integer(len(args[0].name))

@mg_builtin('loop_simple')
def mgb_loop(ctx, args):
    while True:
        ret = args[0].eval_call(ctx, [])
        if ret.value == 0:
            break

__all__ = builtins
