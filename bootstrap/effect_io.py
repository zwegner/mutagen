from syntax import *

@builtin_class('IOHandle', params=[])
class BuiltinIOHandle(BuiltinClass):
    pass

@builtin_class('IOOpenEffect', params=['path'], types=[StrClass], kw_params={'mode': StrClass})
class BuiltinIOOpenEffect(BuiltinClass):
    pass

@builtin_class('IOReadEffect', params=['handle'], types=[BuiltinIOHandle])
class BuiltinIOReadEffect(BuiltinClass):
    pass

@builtin_class('IOWriteEffect', params=['handle', 'bytes'], types=[BuiltinIOHandle, StrClass])
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
    raise ResumeExc(handle)

def handle_read(ctx, effect):
    handle = effect.get_attr(ctx, 'handle')
    raise ResumeExc(String(handle._handle.read(), info=BUILTIN_INFO))

def handle_write(ctx, effect):
    handle = effect.get_attr(ctx, 'handle')
    bytes = effect.get_attr(ctx, 'bytes')
    assert isinstance(bytes, String)
    handle._handle.write(bytes.value)
    raise ResumeExc(NONE)

io_effect_handlers = [
    BuiltinEffectHandler(BuiltinIOOpenEffect, handle_open),
    BuiltinEffectHandler(BuiltinIOReadEffect, handle_read),
    BuiltinEffectHandler(BuiltinIOWriteEffect, handle_write),
]

def wrap_with_io_consumer(ctx, stmt):
    return Consume(stmt, io_effect_handlers)
