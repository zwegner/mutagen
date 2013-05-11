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
    sys.stdout.write(args[0].value)
    return syntax.Nil()

@mg_builtin('len')
def mgb_len(ctx, args):
    return syntax.Integer(len(args[0]))

@mg_builtin('repr')
def mgb_repr(ctx, args):
    arg, = args
    return syntax.String(repr(arg.eval(ctx)))

@mg_builtin('str')
def mgb_str(ctx, args):
    arg, = args
    return syntax.String(str(arg.eval(ctx)))

@mg_builtin('make')
def mgb_make(ctx, args):
    return syntax.Object(args)

@mg_builtin('reduce')
def mgb_reduce(ctx, args):
    fn, start, iter = args
    fn = fn.eval(ctx)
    for i in iter:
        start = fn.eval_call(ctx, [start, i.eval(ctx)])
    return start

@mg_builtin('slice')
def mgb_slice(ctx, args):
    seq, start, end = args
    if isinstance(seq, syntax.String):
        return syntax.String(seq.value[start.value:end.value])
    return syntax.Nil()

@mg_builtin('parse_int')
def mgb_parse_int(ctx, args):
    int_str, base = args
    return Integer(int(int_str, base))

__all__ = builtins
