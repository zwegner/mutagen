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
        builtins[name] = BuiltinFunction(name, builtin_call, info=Info('__builtins__', 0))
        return builtin_call
    return annotate

@mg_builtin([Node])
def mgb_len(ctx, arg):
    return Integer(arg.len(ctx), info=arg)

@mg_builtin([Node])
def mgb_repr(ctx, arg):
    return String(arg.repr(ctx), info=arg)

@mg_builtin([Node])
def mgb_str(ctx, arg):
    return String(arg.str(ctx), info=arg)

@mg_builtin([String])
def mgb_error(ctx, msg):
    msg.error(msg.value, ctx=ctx)

@mg_builtin([Node, Integer, Integer])
def mgb_slice(ctx, seq, start, end):
    if isinstance(seq, String):
        return String(seq.value[start.value:end.value], info=seq)
    elif isinstance(seq, List):
        return List(seq.items[start.value:end.value], info=seq)
    return seq.error('slice on unslicable type %s' % type(seq).__name__, ctx=ctx)

# Python interfacing functions! At some point these would become a security
# hole if we cared about the Python interpreter.
@mg_builtin([String])
def mgb_py_obj_get(ctx, name):
    return PyObject(eval(name.value), info=name)

@mg_builtin([Node])
def mgb_py_wrap(ctx, arg):
    return PyObject(arg, info=arg)

__all__ = builtins
