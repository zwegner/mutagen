filename = 'filename'

class Context:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.syms = {}
    def store(self, name, value):
        self.syms[name] = value
    def load(self, name):
        ctx = self
        while ctx:
            if name in ctx.syms:
                return ctx.syms[name]
            ctx = ctx.parent
        self.print_stack()
        raise Exception('%s not found' % name)
    def assert_true(self, expr):
        if not expr:
            raise Exception()
    def print_stack(self):
        if self.parent:
            self.parent.print_stack()
        print(self.name)

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
def node(argstr='', compare=False):
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

            if hasattr(self, 'setup'):
                self.setup()
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

        # If the compare flag is set, we defer the comparison to the
        # Python object in the value attribute
        if compare:
            def __eq__(self, other):
                return Integer(isinstance(other, type(self)) and
                        self.value == other.value)
            def __ne__(self, other):
                return Integer(not isinstance(other, type(self)) or
                        self.value != other.value)
            def __ge__(self, other):
                assert isinstance(other, Integer)
                return Integer(self.value >= other.value)
            def __gt__(self, other):
                assert isinstance(other, Integer)
                return Integer(self.value > other.value)
            def __le__(self, other):
                assert isinstance(other, Integer)
                return Integer(self.value <= other.value)
            def __lt__(self, other):
                assert isinstance(other, Integer)
                return Integer(self.value < other.value)
            node.__eq__ = __eq__
            node.__ne__ = __ne__
            node.__ge__ = __ge__
            node.__gt__ = __gt__
            node.__le__ = __le__
            node.__lt__ = __lt__

        node.__init__ = __init__
        node.iterate_subtree = iterate_subtree
        node.is_atom = atom

        return node

    return attach

@node()
class Nil(Node):
    def __str__(self):
        return 'Nil'
    def test_truth(self):
        return False

@node('name')
class Identifier(Node):
    def eval(self, ctx):
        return ctx.load(self.name)
    def __str__(self):
        return '%s' % self.name

@node('value', compare=True)
class String(Node):
    def __str__(self):
        return '"%s"' % self.value
    def __getitem__(self, item):
        if isinstance(item, Integer):
            return String(self.value[item.value])
        raise Exception()
    def __add__(self, other):
        assert isinstance(other, String)
        return String(self.value + other.value)
    def __len__(self):
        return len(self.value)

@node('value', compare=True)
class Integer(Node):
    def setup(self):
        self.value = int(self.value)
    def eval(self, ctx):
        return self
    def __str__(self):
        return '%s' % self.value
    def test_truth(self):
        return self.value != 0
    def __not__(self):
        return Integer(self.value == 0)
    def __add__(self, other):
        assert isinstance(other, Integer)
        return Integer(self.value + other.value)

@node('*items')
class List(Node):
    def eval(self, ctx):
        return List([i.eval(ctx) for i in self.items])
    def __str__(self):
        return '[%s]' % ', '.join(str(s) for s in self.items)
    def __iter__(self):
        yield from self.items
    def __getitem__(self, item):
        if isinstance(item, Integer):
            return self.items[item.value]
        raise Exception()
    def __add__(self, other):
        assert isinstance(other, List)
        return List(self.items + other.items)
    def __len__(self):
        return len(self.items)

# HACK: need a dict type
@node('items')
class Object(Node):
    def eval(self, ctx):
        return Object([[k.eval(ctx), v.eval(ctx)] for k, v
            in self.items])
    def __str__(self):
        return '{%s}' % ', '.join('%s:%s' % (k, v) for k, v
                in self.items)

@node('&method, &self')
class BoundMethod(Node):
    def eval(self, ctx):
        return self
    def eval_call(self, ctx, args):
        method = self.method.eval(ctx)
        args = [self.self] + args
        return method.eval_call(ctx, args)
    def __str__(self):
        return '%s.%s' % (self.self, self.method)

@node('type, &rhs')
class UnaryOp(Node):
    def eval(self, ctx):
        rhs = self.rhs.eval(ctx)
        if self.type == 'not':
            return rhs.__not__()
        raise Exception()
    def __str__(self):
        return '(%s %s)' % (self.type, self.rhs)

@node('type, &lhs, &rhs')
class BinaryOp(Node):
    def eval(self, ctx):
        lhs = self.lhs.eval(ctx)
        rhs = self.rhs.eval(ctx)
        # This is a bit of an abuse of Python operator overloading! Oh well...
        operator = {
            '==': 'eq',
            '!=': 'ne',
            '>':  'gt',
            '>=': 'ge',
            '<':  'lt',
            '<=': 'le',
            '+':  'add',
        }[self.type]
        return getattr(lhs, '__%s__' % operator)(rhs)
    def __str__(self):
        return '(%s %s %s)' % (self.lhs, self.type, self.rhs)

@node('&obj, attr')
class GetAttr(Node):
    def eval(self, ctx):
        # HACK: no real dictionaries
        def get(obj, attr):
            items = obj.items
            results = [v for k, v in items if k.value == attr]
            return results[0] if results else None
        obj = self.obj.eval(ctx)
        item = get(obj, self.attr)
        if item is None:
            item = BoundMethod(GetAttr(get(obj, '__class__'), self.attr), obj)
        return item
    def __str__(self):
        return '%s.%s' % (self.obj, self.attr)

@node('&obj, &item')
class GetItem(Node):
    def eval(self, ctx):
        # HACK: no real dictionaries
        obj = self.obj.eval(ctx)
        item = self.item.eval(ctx)
        return obj[item]
    def __str__(self):
        return '%s[%s]' % (self.obj, self.item)

@node('name, &rhs')
class Assignment(Node):
    def eval(self, ctx):
        value = self.rhs.eval(ctx)
        ctx.store(self.name, value)
        return value
    def __str__(self):
        return '%s = %s' % (self.name, self.rhs)

@node('&expr, *if_stmts, *else_stmts')
class IfElse(Node):
    def eval(self, ctx):
        expr = self.expr.eval(ctx).test_truth()
        block = self.if_stmts if expr else self.else_stmts
        value = Nil()
        for stmt in block:
            value = stmt.eval(ctx)
        return value
    def __str__(self):
        else_block = ''
        if self.else_stmts:
            else_block = ' else {%s}' % '\n'.join(str(s) for s in self.else_stmts)
        if_block = '{%s}' % '\n'.join(str(s) for s in self.if_stmts)
        return 'if %s %s%s' % (self.expr, if_block, else_block)

@node('iter, &expr, *body')
class For(Node):
    def eval(self, ctx):
        expr = self.expr.eval(ctx)
        for i in expr:
            ctx.store(self.iter, i)
            for stmt in self.body:
                stmt.eval(ctx)
        return Nil()
    def __str__(self):
        body = '{%s}' % '\n'.join(str(s) for s in self.body)
        return 'for %s in %s:\n %s' % (self.iter, self.expr, body)

@node('&expr, *body')
class While(Node):
    def eval(self, ctx):
        while self.expr.eval(ctx).test_truth():
            for stmt in self.body:
                stmt.eval(ctx)
        return Nil()
    def __str__(self):
        body = '{%s}' % '\n'.join(str(s) for s in self.body)
        return 'while %s:\n %s' % (self.expr, body)

@node('&fn, *args')
class Call(Node):
    def eval(self, ctx):
        fn = self.fn.eval(ctx)
        args = [a.eval(ctx) for a in self.args]
        return fn.eval_call(ctx, args)
    def __str__(self):
        return '%s(%s)' % (self.fn, ', '.join(str(s) for s in self.args))

@node('name, params, *block')
class Function(Node):
    def eval(self, ctx):
        return self
    def eval_call(self, ctx, args):
        ret = Nil()
        child_ctx = Context(self.name, ctx)
        for p, a in zip(self.params, args):
            child_ctx.store(p, a)
        for expr in self.block:
            ret = expr.eval(child_ctx)
        return ret
    def __str__(self):
        return 'def %s[%s]{%s}' % (self.name, ', '.join(str(s)
            for s in self.params), '\n'.join(str(s) for s in self.block))

@node('name, &fn')
class BuiltinFunction(Node):
    def eval_call(self, ctx, args):
        child_ctx = Context(self.name, ctx)
        return self.fn(child_ctx, args)
    def __str__(self):
        return '<builtin %s>' % self.name

@node('name')
class Import(Node):
    def eval(self, ctx):
        raise Exception()
    def __str__(self):
        return 'import %s' % self.name
