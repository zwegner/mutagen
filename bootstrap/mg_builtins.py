import sys

from syntax import *

builtins = {}

def mg_builtin(arg_types):
    def annotate(fn):
        def builtin_call(obj, ctx, args):
            if arg_types is not None:
                if len(args) != len(arg_types):
                    obj.error('incorrect number of arguments to builtin %s' %
                            obj.name, ctx=ctx)
                for a, t in zip(args, arg_types):
                    if not isinstance(a, t):
                        obj.error('bad argument to builtin %s, expected %s, got %s' %
                                (obj.name, t.__name__, type(a).__name__), ctx=ctx)
            ctx.current_node = obj
            return fn(ctx, *args)

        name = fn.__name__.replace('mgb_', '')
        builtins[name] = BuiltinFunction(name, builtin_call, info=builtin_info)
        return builtin_call
    return annotate

# XXX hack
@mg_builtin([String])
def mgb_read_file(ctx, path):
    with open(path.value) as f:
        return String(f.read(), info=path)

# XXX hack
@mg_builtin([String, List])
def mgb_write_binary_file(ctx, path, data):
    if any(not isinstance(i, Integer) or not 0 <= i.value < 256 for i in data):
        return data.error('data must be an array of integers from 0-255', ctx=ctx)
    data = bytes(i.value for i in data)
    with open(path.value, 'wb') as f:
        f.write(data)
    return None_(info=path)

@mg_builtin([String])
def mgb_putstr(ctx, arg):
    sys.stdout.write(arg.value)
    return None_(info=arg)

@mg_builtin([Node])
def mgb_len(ctx, arg):
    return Integer(arg.len(ctx), info=arg)

@mg_builtin([Node])
def mgb_repr(ctx, arg):
    return String(arg.repr(ctx), info=arg)

@mg_builtin([String])
def mgb_error(ctx, msg):
    msg.error(msg.value, ctx=ctx)

@mg_builtin(None)
def mgb_slice(ctx, seq, *args):
    args = [a.value if a is not None and not isinstance(a, None_) else None
            for a in args]
    [start, stop, step] = [None, None, None]
    if len(args) == 1:
        [stop] = args
    elif len(args) == 2:
        [start, stop] = args
    elif len(args) == 3:
        [start, stop, step] = args
    else:
        seq.error('expected 1-3 arguments to slice, got %s' % len(args), ctx=ctx)
    if isinstance(seq, String):
        return String(seq.value[start:stop:step], info=seq)
    elif isinstance(seq, List):
        return List(seq.items[start:stop:step], info=seq)
    return seq.error('slice on unsliceable type %s' % type(seq).__name__, ctx=ctx)

# XXX remove this, just temporarily added to test stuff that should fail.
# Need to work out semantics of any exceptions/error handling first.
@mg_builtin(None)
def mgb_assert_call_fails(ctx, fn, *args):
    try:
        fn.eval_call(ctx, [a.eval(ctx) for a in args])
    except ProgramError as e:
        return None_(info=fn)
    fn.error('did not throw error', ctx=ctx)

# Add builtin classes
builtins['str'] = StrClass
builtins['int'] = IntClass
builtins['bool'] = BoolClass
builtins['list'] = ListClass
builtins['dict'] = DictClass
builtins['type'] = TypeClass

__all__ = builtins
