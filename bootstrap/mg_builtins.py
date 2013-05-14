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

@mg_builtin('read_file')
def mgb_read_file(ctx, args):
    path, = args
    with open(path.value) as f:
        return syntax.String(f.read())

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
    return syntax.String(repr(arg))

@mg_builtin('str')
def mgb_str(ctx, args):
    arg, = args
    return syntax.String(str(arg))

@mg_builtin('make')
def mgb_make(ctx, args):
    return syntax.Object(args)

@mg_builtin('reduce')
def mgb_reduce(ctx, args):
    fn, start, iter = args
    for i in iter:
        start = fn.eval_call(ctx, [start, i.eval(ctx)])
    return start

@mg_builtin('slice')
def mgb_slice(ctx, args):
    seq, start, end = args
    if isinstance(seq, syntax.String):
        return syntax.String(seq.value[start.value:end.value])
    elif isinstance(seq, syntax.List):
        return syntax.List(seq.items[start.value:end.value])
    return syntax.Nil()

@mg_builtin('parse_int')
def mgb_parse_int(ctx, args):
    int_str, base = args
    assert (isinstance(int_str, syntax.String) and
            isinstance(base, syntax.Integer))
    return syntax.Integer(int(int_str.value, base.value))

@mg_builtin('str_upper')
def mgb_str_upper(ctx, args):
    arg, = args
    assert isinstance(arg, syntax.String)
    return syntax.String(arg.value.upper())

__all__ = builtins
