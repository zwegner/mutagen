import collections
import copy
import sys

from .syntax import *

def create_builtin_function(name, fn, arg_types):
    def builtin_call(ctx, args, kwargs):
        assert kwargs == {}
        # HACK: inspect context to get the call site of this function for better errors
        info = ctx.current_node
        if arg_types is not None:
            if len(args) != len(arg_types):
                info.error('incorrect number of arguments to builtin %s' %
                        name, ctx=ctx)
            for a, t in zip(args, arg_types):
                if not isinstance(a, t):
                    info.error('bad argument to builtin %s, expected %s, got %s' %
                            (name, t.__name__, type(a).__name__), ctx=ctx)
        return fn(ctx, *args)
    return BuiltinFunction(name, builtin_call, info=BUILTIN_INFO)

def mg_builtin(arg_types):
    def annotate(fn):
        name = fn.__name__.replace('mgb_', '')
        builtins[name] = create_builtin_function(name, fn, arg_types)
        return None
    return annotate

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
    args = [a.value if a is not NONE else None for a in args]
    [start, stop, step] = [None] * 3
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

@mg_builtin([Node])
def mgb_hash(ctx, arg):
    return Integer(hash(arg), info=arg)

@mg_builtin([Node, String])
def mgb_getattr(ctx, arg, attr):
    # HACK: inspect context to get the call site of this function for better errors
    info = ctx.current_node
    return get_attr(ctx, arg, attr.value, info=info)

@mg_builtin([Node, String])
def mgb_hasattr(ctx, arg, attr):
    return Boolean(get_attr(ctx, arg, attr.value, raise_errors=False) is not None, info=arg)

@mg_builtin([Node])
def mgb_dir(ctx, obj):
    if isinstance(obj, Class):
        obj = obj.cls
    if isinstance(obj, Object):
        return List(list(obj.items), info=obj)
    return obj.error('Unsupported type for dir()', ctx=ctx)

@mg_builtin([String, Class, Dict, Params])
def mgb_hacky_class_from_base_and_new_attrs(ctx, name, base, attrs, params):
    base = copy.copy(base)
    base.params = params
    base.cls = copy.copy(base.cls)
    base.cls.items = copy.copy(base.cls.items)
    base.cls.items.update(attrs.items)
    return base

# HACK
@mg_builtin([String, Node, Node])
def mgb_KeywordParam(ctx, name, type, default):
    return KeywordParam(name.value, type, default, info=name)

# HACK
@mg_builtin([List, List, Node, List, Node])
def mgb_Parameters(ctx, names, types, var_params, kw_params, kw_var_params):
    info = names
    assert all(isinstance(name, String) for name in names.items)
    names = [name.value for name in names.items]
    types = types.items

    assert isinstance(var_params, String) or var_params is NONE
    var_params = var_params.value if isinstance(var_params, String) else None

    assert all(isinstance(kwparam, KeywordParam) for kwparam in kw_params)
    kw_params = kw_params.items

    assert isinstance(kw_var_params, String) or kw_var_params is NONE
    kw_var_params = kw_var_params.value if isinstance(kw_var_params, String) else None

    params = Params(names, types, var_params, kw_params, kw_var_params, info=info)

    # HACK--this normally happens during specialization. Assume types are already evaluated
    params.type_evals = types
    params.keyword_evals = {p.name: p for p in kw_params}

    return params

# XXX nasty. Directly modifies the class, amongst other bad things
@mg_builtin([Class, Class])
def mgb_hacky_inherit_from(ctx, parent, cls):
    for attr in parent.cls.items:
        if attr not in cls.cls.items:
            cls.cls.items[attr] = parent.cls.items[attr]
    cls.cls.items[String('__parent__', info=BUILTIN_INFO)] = parent
    return cls

# XXX remove this, just temporarily added to test stuff that should fail.
# Need to work out semantics of any exceptions/error handling first.
@mg_builtin(None)
def mgb_assert_call_fails(ctx, fn, *args):
    try:
        fn.eval_call(ctx, args, {})
    except ProgramError as e:
        return NONE
    fn.error('did not throw error', ctx=ctx)

# XXX remove this, temporary shim to speed up parsing
@mg_builtin([String])
def mgb_re_compile_match(ctx, regex):
    import re
    match = re.compile(regex.value).match
    def match_internal(ctx, text, start):
        # HACK: inspect context to get the call site of this function for better errors
        info = ctx.current_node
        m = match(text.value, start.value)
        if not m:
            return NONE
        type = String(m.lastgroup, info=info)
        start = Integer(m.start(), info=info)
        end = Integer(m.end(), info=info)
        return List([type, start, end], info=info)
    return create_builtin_function('re_compile_match', match_internal, [String, Integer])
