import sys

import syntax

builtins = {}

builtin_info = syntax.Info('__builtins__', 0)
def mg_builtin(name, arg_types):
    def annotate(fn):
        def builtin_call(ctx, args):
            if arg_types is not None:
                assert len(args) == len(arg_types)
                for a, t in zip(args, arg_types):
                    assert isinstance(a, t)
            return fn(ctx, *args)

        builtins[name] = syntax.BuiltinFunction(name, builtin_call, info=builtin_info)
        return builtin_call
    return annotate

# TODO: error checking!

@mg_builtin('read_file', [syntax.String])
def mgb_read_file(ctx, path):
    with open(path.value) as f:
        return syntax.String(f.read(), info=builtin_info)

@mg_builtin('putchar', [syntax.String])
def mgb_putchar(ctx, arg):
    sys.stdout.write(arg.value)
    return syntax.Nil(info=builtin_info)

@mg_builtin('len', [syntax.Node])
def mgb_len(ctx, arg):
    return syntax.Integer(arg.len(ctx), info=arg)

@mg_builtin('repr', [syntax.Node])
def mgb_repr(ctx, arg):
    return syntax.String(arg.repr(ctx), info=arg)

@mg_builtin('str', [syntax.Node])
def mgb_str(ctx, arg):
    return syntax.String(arg.str(ctx), info=arg)

@mg_builtin('make', [syntax.Dict])
def mgb_make(ctx, arg):
    return syntax.Object(arg, info=builtin_info)

@mg_builtin('error', [syntax.String])
def mgb_error(ctx, msg):
    msg.error(msg.value, ctx=ctx)

@mg_builtin('reduce', [syntax.Function, syntax.Node, syntax.Node])
def mgb_reduce(ctx, fn, start, iter):
    for i in iter.iter(ctx):
        start = fn.eval_call(ctx, [start, i.eval(ctx)])
    return start

@mg_builtin('slice', [syntax.Node, syntax.Integer, syntax.Integer])
def mgb_slice(ctx, seq, start, end):
    if isinstance(seq, syntax.String):
        return syntax.String(seq.value[start.value:end.value], info=builtin_info)
    elif isinstance(seq, syntax.List):
        return syntax.List(seq.items[start.value:end.value], info=builtin_info)
    return syntax.Nil(info=builtin_info)

@mg_builtin('parse_int', [syntax.String, syntax.Integer])
def mgb_parse_int(ctx, int_str, base):
    return syntax.Integer(int(int_str.value, base.value), info=builtin_info)

@mg_builtin('str_upper', [syntax.String])
def mgb_str_upper(ctx, arg):
    return syntax.String(arg.value.upper(), info=builtin_info)

__all__ = builtins
