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
    sys.stdout.write(args[0].name + '\n')
    return syntax.Nil()

@mg_builtin('len')
def mgb_len(ctx, args):
    return syntax.Integer(len(args[0].name))

@mg_builtin('str')
def mgb_str(ctx, args):
    arg, = args
    return syntax.String(str(arg.eval(ctx)))

@mg_builtin('make')
def mgb_make(ctx, args):
    items = dict(args)
    return syntax.Object(items)

@mg_builtin('map')
def mgb_map(ctx, args):
    fn, iter = args
    fn = fn.eval(ctx)
    l = []
    for i in iter:
        l += [fn.eval_call(ctx, [i.eval(ctx)])]
    return syntax.List(l)

@mg_builtin('reduce')
def mgb_reduce(ctx, args):
    fn, start, iter = args
    fn = fn.eval(ctx)
    for i in iter:
        start = fn.eval_call(ctx, [start, i.eval(ctx)])
    return start

__all__ = builtins
