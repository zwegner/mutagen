filename = 'filename'

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
    def eval(self, ctx):
        return self

    def add_use(self, edge):
        assert edge not in self.uses
        self.uses.append(edge)

    def remove_use(self, edge):
        self.uses.remove(edge)

    def forward(self, new_value):
        for edge in self.uses:
            edge.value = new_value
            new_value.add_use(edge)

    def __str__(self):
        assert False

# Edge class represents the use of a Node by another. This allows us to use
# value forwarding and such. It is used like so:
#
# self.expr = Edge(expr) # done implicitly by each class' constructor
# value = self.expr()
# self.expr.set(new_value)
class Edge:
    def __init__(self, value):
        self.value = value
        value.add_use(self)

    def __call__(self):
        return self.value

    def set(self, value):
        self.value.remove_use(self)
        self.value = value
        value.add_use(self)

    def __str__(self):
        print(self.value)
        assert False

ARG_REG, ARG_EDGE, ARG_EDGE_LIST = list(range(3))
arg_map = {'&': ARG_EDGE, '*': ARG_EDGE_LIST}

# Weird decorator: a given arg string represents a standard form for arguments
# to Node subclasses. We use these notations:
# op, &expr, *explist
# op -> normal attribute
# &expr -> edge attribute, will create an Edge object (used for linking to other Nodes)
# *explist -> python list of edges
def node(argstr=''):
    args = [a.strip() for a in argstr.split(',') if a.strip()]
    new_args = []
    for a in args:
        if a[0] in arg_map:
            new_args.append((arg_map[a[0]], a[1:]))
        else:
            new_args.append((ARG_REG, a))
    args = new_args

    atom = not any(a[0] != ARG_REG for a in args)

    # Decorators must return a function. This adds __init__ and some other methods
    # to a Node subclass
    def attach(node):
        def __init__(self, *iargs):
            assert len(iargs) == len(args), 'bad args, expected %s(%s)' % (node.__name__, argstr)

            for (arg_type, arg_name), v in zip(args, iargs):
                #if arg_type == ARG_EDGE:
                #    setattr(self, arg_name, Edge(v) if v is not None else None)
                #elif arg_type == ARG_EDGE_LIST:
                #    setattr(self, arg_name, [Edge(item) for item in v])
                #else:
                #    setattr(self, arg_name, v)
                setattr(self, arg_name, v)

            self.uses = []

        def iterate_subtree(self):
            yield self
            for (arg_type, arg_name) in args:
                if arg_type == ARG_EDGE:
                    edge = getattr(self, arg_name)
                    yield from edge().iterate_subtree()
                elif arg_type in {ARG_EDGE_LIST, ARG_BLOCK}:
                    for edge in getattr(self, arg_name):
                        yield from edge().iterate_subtree()

        node.__init__ = __init__
        node.iterate_subtree = iterate_subtree
        node.is_atom = atom

        return node

    return attach

@node()
class Nil(Node):
    def __str__(self):
        return 'Nil'

@node('name')
class Identifier(Node):
    def eval(self, ctx):
        return ctx.load(self.name)
    def __str__(self):
        return '<%s>' % self.name

@node('name')
class String(Node):
    def __str__(self):
        return '"%s"' % self.name

@node('value')
class Integer(Node):
    def eval(self, ctx):
        return self
    def __str__(self):
        return '%s' % self.value

@node('*items')
class List(Node):
    def eval(self, ctx):
        return List([i.eval(ctx) for i in self.items])
    def __str__(self):
        return '[%s]' % ', '.join(str(s) for s in self.items)
    def __iter__(self):
        yield from self.items

# HACK: need a dict type
@node('items')
class Object(Node):
    def eval(self, ctx):
        return Object({k.eval(ctx): v.eval(ctx) for k, v
            in self.items.items()})
    def __str__(self):
        return '{%s}' % ', '.join('%s:%s' % (k, v) for k, v
                in self.items.items())

@node('&obj, attr')
class GetAttr(Node):
    def eval(self, ctx):
        # HACK: no real dictionaries
        items = self.obj.eval(ctx).items
        item, = [v for k, v in items.items() if k.name == self.attr.name]
        return item
    def __str__(self):
        return '%s.%s' % (self.obj, self.attr)

@node('name, &rhs')
class Assignment(Node):
    def eval(self, ctx):
        value = self.rhs.eval(ctx)
        ctx.store(self.name, value)
        return value
    def __str__(self):
        return '%s = %s' % (self.name, self.rhs)

@node('&fn, *args')
class Call(Node):
    def eval(self, ctx):
        fn = self.fn.eval(ctx)
        args = [a.eval(ctx) for a in self.args]
        return fn.eval_call(ctx, args)
    def __str__(self):
        return '%s(%s)' % (self.fn, ', '.join(str(s) for s in self.args))

@node('params, *block')
class Function(Node):
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

@node('name, &fn')
class BuiltinFunction(Node):
    def eval_call(self, ctx, args):
        return self.fn(ctx, args)
    def __str__(self):
        return '<builtin %s>' % self.name

@node('name')
class Import(Node):
    def eval(self, ctx):
        raise Exception()
    def __str__(self):
        return 'import %s' % self.name
