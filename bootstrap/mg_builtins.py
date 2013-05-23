import sys

import syntax

builtins = {}

builtin_info = syntax.Info('__builtins__', 0)
def mg_builtin(name):
    def annotate(fn):
        builtins[name] = syntax.BuiltinFunction(name, fn, info=builtin_info)
        return fn
    return annotate

# TODO: error checking!

@mg_builtin('read_file')
def mgb_read_file(ctx, args):
    path, = args
    with open(path.value) as f:
        return syntax.String(f.read(), info=builtin_info)

@mg_builtin('putchar')
def mgb_putchar(ctx, args):
    sys.stdout.write(args[0].value)
    return syntax.Nil(info=builtin_info)

@mg_builtin('len')
def mgb_len(ctx, args):
    return syntax.Integer(len(args[0]), info=builtin_info)

@mg_builtin('repr')
def mgb_repr(ctx, args):
    arg, = args
    return syntax.String(repr(arg), info=builtin_info)

@mg_builtin('str')
def mgb_str(ctx, args):
    arg, = args
    return syntax.String(str(arg), info=builtin_info)

@mg_builtin('make')
def mgb_make(ctx, args):
    return syntax.Object(args, info=builtin_info)

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
        return syntax.String(seq.value[start.value:end.value], info=builtin_info)
    elif isinstance(seq, syntax.List):
        return syntax.List(seq.items[start.value:end.value], info=builtin_info)
    return syntax.Nil(info=builtin_info)

@mg_builtin('parse_int')
def mgb_parse_int(ctx, args):
    int_str, base = args
    assert (isinstance(int_str, syntax.String) and
            isinstance(base, syntax.Integer))
    return syntax.Integer(int(int_str.value, base.value), info=builtin_info)

@mg_builtin('str_upper')
def mgb_str_upper(ctx, args):
    arg, = args
    assert isinstance(arg, syntax.String)
    return syntax.String(arg.value.upper(), info=builtin_info)

__all__ = builtins
