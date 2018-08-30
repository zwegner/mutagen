import sys

from syntax import *

@builtin_class('IOHandle', params=[])
class BuiltinIOHandle(BuiltinClass):
    pass

@builtin_class('IOOpenEffect', params=['path'], types=[StrClass], kw_params={'mode': StrClass})
class BuiltinIOOpenEffect(BuiltinClass):
    pass

@builtin_class('IOCloseEffect', params=['handle'], types=[BuiltinIOHandle])
class BuiltinIOCloseEffect(BuiltinClass):
    pass

@builtin_class('IOReadEffect', params=['handle'], types=[BuiltinIOHandle])
class BuiltinIOReadEffect(BuiltinClass):
    pass

@builtin_class('IOWriteEffect', params=['handle', 'data'], types=[BuiltinIOHandle, NONE])
class BuiltinIOWriteEffect(BuiltinClass):
    pass

@node('&type, fn')
class BuiltinEffectHandler(EffectHandler):
    def handle_effect(self, ctx, effect):
        self.fn(ctx, effect)
    def repr(self, ctx):
        return '<builtin effect handler %s>' % self.fn.__name__

def handle_open(ctx, effect):
    path = effect.get_attr(ctx, 'path')
    mode = effect.get_attr(ctx, 'mode')
    mode = mode.value if mode is not NONE else 'r'
    assert isinstance(path, String)
    handle = BuiltinIOHandle.eval_call(ctx, [], {})
    handle._handle = open(path.value, mode)
    handle._mode = mode
    raise ResumeExc(handle)

def handle_close(ctx, effect):
    handle = effect.get_attr(ctx, 'handle')
    handle._handle.close()
    raise ResumeExc(NONE)

def handle_read(ctx, effect):
    handle = effect.get_attr(ctx, 'handle')
    data = handle._handle.read()
    if 'b' in handle._mode:
        result = List([Integer(i, info=BUILTIN_INFO) for i in data], info=BUILTIN_INFO)
    else:
        result = String(data, info=BUILTIN_INFO)
    raise ResumeExc(result)

def handle_write(ctx, effect):
    handle = effect.get_attr(ctx, 'handle')
    data = effect.get_attr(ctx, 'data')
    if 'b' in handle._mode:
        # XXX remove when we have a proper bytes class
        if not isinstance(data, List) or any(not isinstance(i, Integer) or
                not 0 <= i.value < 256 for i in data):
            return data.error('data must be an array of integers from 0-255', ctx=ctx)
        data = bytes(i.value for i in data)
    else:
        if not isinstance(data, String):
            return data.error('data must be a string', ctx=ctx)
        data = data.value
    handle._handle.write(data)
    raise ResumeExc(NONE)

# Create builtin functions to get default handles (stdin etc.). This is kinda ridiculous,
# we can't create objects yet because we don't have a context, so we can't store these
# directly in the builtins. And we do some dumb caching just for the hell of it.
for name, stream in [['IN', sys.stdin], ['OUT', sys.stdout], ['ERR', sys.stderr]]:
    name = '_GET_STD%s_HANDLE' % name
    def capture(stream):
        handle = None
        def create(ctx, args, kwargs):
            nonlocal handle
            if not handle:
                handle = BuiltinIOHandle.eval_call(ctx, [], {})
                handle._handle = stream
                handle._mode = ''
            return handle
        return BuiltinFunction(name, create, info=BUILTIN_INFO)
    builtins[name] = capture(stream)

io_effect_handlers = [
    BuiltinEffectHandler(BuiltinIOOpenEffect, handle_open),
    BuiltinEffectHandler(BuiltinIOCloseEffect, handle_close),
    BuiltinEffectHandler(BuiltinIOReadEffect, handle_read),
    BuiltinEffectHandler(BuiltinIOWriteEffect, handle_write),
]

def wrap_with_io_consumer(ctx, stmt):
    return Consume(stmt, io_effect_handlers)
