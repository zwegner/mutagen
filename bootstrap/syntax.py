class Context:
    def __init__(self, parent):
        self.parent = parent
        self.syms = {}
    def store(self, name, value):
        self.syms[name] = value
    def load(self, name):
        if name in self.syms:
            return self.syms[name]
        if not self.parent:
            raise Exception('%s not found' % name)
        return self.parent.load(name)
    def assert_true(self, expr):
        if not expr:
            raise Exception()

class Node:
    pass

class Nil(Node):
    def __str__(self):
        return 'Nil'

class Identifier(Node):
    def __init__(self, name):
        self.name = name
    def eval(self, ctx):
        return ctx.load(self.name)
    def __str__(self):
        return '<%s>' % self.name

class String(Node):
    def __init__(self, name):
        self.name = name
    def eval(self, ctx):
        return self
    def __str__(self):
        return self.name
    def __iter__(self):
        for i in self.items:
            yield i

class Integer(Node):
    def __init__(self, value):
        self.value = value
    def eval(self, ctx):
        return self.value
    def __str__(self):
        return '%s' % self.value

class List(Node):
    def __init__(self, items):
        self.items = items
    def eval(self, ctx):
        return self
    def __str__(self):
        return '[%s]' % ', '.join(str(s) for s in self.items)
    def __iter__(self):
        for i in self.items:
            yield i

class Assignment(Node):
    def __init__(self, name, rhs):
        self.name = name.name
        self.rhs = rhs
    def eval(self, ctx):
        value = self.rhs.eval(ctx)
        ctx.store(self.name, value)
        return value
    def __str__(self):
        return '%s = %s' % (self.name, self.rhs)

class Call(Node):
    def __init__(self, name, args):
        self.name = name
        self.args = args
    def eval(self, ctx):
        fn = self.name.eval(ctx)
        args = [a.eval(ctx) for a in self.args]
        return fn.eval_call(ctx, args)
    def __str__(self):
        return '%s(%s)' % (self.name, ', '.join(str(s) for s in self.args))

class Function(Node):
    def __init__(self, params, block):
        self.params = params
        self.block = block
    def eval(self, ctx):
        return self
    def eval_call(self, ctx, args):
        ret = Nil()
        child_ctx = Context(ctx)
        for p, a in zip(self.params, args):
            child_ctx.store(p.name, a)
        for expr in self.block:
            ret = expr.eval(child_ctx)
        return ret
    def __str__(self):
        return '[%s]{%s}' % (', '.join(str(s) for s in self.params), '\n'.join(str(s) for s in self.block))

class BuiltinFunction(Node):
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn
    def eval_call(self, ctx, args):
        return self.fn(ctx, args)
    def __str__(self):
        return '<builtin %s>' % self.name

class Import(Node):
    def __init__(self, name):
        self.name = name
    def eval(self, ctx):
        raise Exception()
    def __str__(self):
        return 'import %s' % self.name
