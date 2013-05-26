import sys

filename = 'filename'

# This is a list, since order matters--backslashes must come first!
inv_str_escapes = [
    ['\\', '\\\\'],
    ['\n', '\\n'],
    ['\t', '\\t'],
    ['\b', '\\x08'], # HACK
]

class Context:
    def __init__(self, name, node, global_ctx, parent):
        self.name = name
        self.node = node
        self.global_ctx = global_ctx
        self.parent = parent
        self.syms = {}
    def store(self, name, value):
        self.syms[name] = value
    def load(self, node, name):
        ctx = self
        while ctx:
            if name in ctx.syms:
                return ctx.syms[name]
            ctx = ctx.global_ctx
        node.error('identifier %s not found' % name, ctx=self)
    def print_stack(self):
        if self.parent:
            self.parent.print_stack()
        if self.node:
            info = self.node.info
            print(' at %s, %s, line %s' % (self.node.name, info.filename, info.lineno), file=sys.stderr)
        else:
            print('in module %s' % self.name, file=sys.stderr)

# Info means basically filename/line number, used for reporting errors
class Info:
    def __init__(self, filename, lineno):
        self.filename = filename
        self.lineno = lineno

class Node:
    def eval(self, ctx):
        return self
    def eval_gen(self, ctx):
        self.eval(ctx)
        return []
    def error(self, msg, ctx=None):
        if ctx:
            ctx.print_stack()
        print('%s(%i): %s' % (self.info.filename, self.info.lineno, msg), file=sys.stderr)
        sys.exit(1)
    def __eq__(self, other):
        self.error('__eq__ unimplemented for type %s' % type(self))
    def __ne__(self, other):
        return Integer(not self.__eq__(other).value, info=self)
    def __str__(self):
        assert False
    def __repr__(self):
        assert False
    def len(self, ctx):
        self.error('__len__ unimplemented for %s' % type(self), ctx=ctx)
    def str(self, ctx):
        return self.repr(ctx)
    def repr(self, ctx):
        self.error('__repr__ unimplemented for %s' % type(self), ctx=ctx)

ARG_REG, ARG_EDGE, ARG_EDGE_LIST = list(range(3))
arg_map = {'&': ARG_EDGE, '*': ARG_EDGE_LIST}

# Weird decorator: a given arg string represents a standard form for arguments
# to Node subclasses. We use these notations:
# op, &expr, *explist
# op -> normal attribute
# &expr -> edge attribute, used for linking to other Nodes
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

    # Decorators must return a function. This adds __init__ and some other methods
    # to a Node subclass
    def attach(node):
        def __init__(self, *iargs, info=None):
            assert len(iargs) == len(args), 'bad args, expected %s(%s)' % (node.__name__, argstr)

            for (arg_type, arg_name), v in zip(args, iargs):
                setattr(self, arg_name, v)

            if info is None:
                for (arg_type, arg_name), v in zip(args, iargs):
                    if arg_type == ARG_EDGE:
                        info = v.info
                        break
            if isinstance(info, Node):
                info = info.info
            self.info = info
            assert self.info

            if hasattr(self, 'setup'):
                self.setup()

        def iterate_subtree(self):
            yield self
            for (arg_type, arg_name) in args:
                if arg_type == ARG_EDGE:
                    edge = getattr(self, arg_name)
                    yield from edge.iterate_subtree()
                elif arg_type == ARG_EDGE_LIST:
                    for edge in getattr(self, arg_name):
                        yield from edge.iterate_subtree()

        # If the compare flag is set, we defer the comparison to the
        # Python object in the value attribute
        if compare:
            def __eq__(self, other):
                return Integer(isinstance(other, type(self)) and
                        self.value == other.value, info=self)
            def __ne__(self, other):
                return Integer(not isinstance(other, type(self)) or
                        self.value != other.value, info=self)
            def __ge__(self, other):
                if not isinstance(other, type(self)):
                    self.error('uncomparable types: %s, %s' % (type(self), type(other)))
                return Integer(self.value >= other.value, info=self)
            def __gt__(self, other):
                if not isinstance(other, type(self)):
                    self.error('uncomparable types: %s, %s' % (type(self), type(other)))
                return Integer(self.value > other.value, info=self)
            def __le__(self, other):
                if not isinstance(other, type(self)):
                    self.error('uncomparable types: %s, %s' % (type(self), type(other)))
                return Integer(self.value <= other.value, info=self)
            def __lt__(self, other):
                if not isinstance(other, type(self)):
                    self.error('uncomparable types: %s, %s' % (type(self), type(other)))
                return Integer(self.value < other.value, info=self)
            def __hash__(self):
                return self.value.__hash__()
            node.__eq__ = __eq__
            node.__ne__ = __ne__
            node.__ge__ = __ge__
            node.__gt__ = __gt__
            node.__le__ = __le__
            node.__lt__ = __lt__
            node.__hash__ = __hash__

        node.__init__ = __init__
        node.iterate_subtree = iterate_subtree
        return node

    return attach

def block_str(block, ctx):
    block = [s.repr(ctx) for s in block]
    block = ['\n    '.join(s for s in b.splitlines()) for b in block]
    return ':\n    %s' % ('\n    '.join(block))

@node()
class Nil(Node):
    def repr(self, ctx):
        return 'Nil'
    def __eq__(self, other):
        return Integer(isinstance(other, Nil), info=self)
    def __ne__(self, other):
        return Integer(not isinstance(other, Nil), info=self)
    def test_truth(self):
        return False
    def __hash__(self):
        return 0

@node('name')
class Identifier(Node):
    def eval(self, ctx):
        return ctx.load(self, self.name)
    def repr(self, ctx):
        return '%s' % self.name

@node('value', compare=True)
class String(Node):
    def str(self, ctx):
        return self.value
    def repr(self, ctx):
        value = self.value
        for k, v in inv_str_escapes:
            value = value.replace(k, v)
        has_quotes = [x in self.value for x in ['\'', '"']]
        if has_quotes[0] and not has_quotes[1]:
            quote = '"'
            value = value.replace('"', '\\"')
        else:
            quote = '\''
            value = value.replace("'", "\\'")
        return '%s%s%s' % (quote, value, quote)
    def __iter__(self):
        for v in self.value:
            yield String(v, info=self)
    def __getitem__(self, item):
        if isinstance(item, Integer):
            return String(self.value[item.value], info=self)
        self.error('bad arg for getitem: %s' % item)
    def __add__(self, other):
        if not isinstance(other, String):
            self.error('bad type for str.add: %s' % type(other))
        return String(self.value + other.value, info=self)
    def len(self, ctx):
        return len(self.value)

@node('value', compare=True)
class Integer(Node):
    def setup(self):
        self.value = int(self.value)
    def eval(self, ctx):
        return self
    def repr(self, ctx):
        return '%s' % self.value
    def test_truth(self):
        return self.value != 0
    def __not__(self):
        return Integer(self.value == 0, info=self)
    def __add__(self, other):
        if not isinstance(other, Integer):
            self.error('bad type for int.add: %s' % type(other))
        return Integer(self.value + other.value, info=self)
    def __sub__(self, other):
        if not isinstance(other, Integer):
            self.error('bad type for int.sub: %s' % type(other))
        return Integer(self.value - other.value, info=self)

@node('*items')
class List(Node):
    def eval(self, ctx):
        return List([i.eval(ctx) for i in self.items], info=self)
    def repr(self, ctx):
        return '[%s]' % ', '.join(s.repr(ctx) for s in self.items)
    def __iter__(self):
        yield from self.items
    def __getitem__(self, item):
        if isinstance(item, Integer):
            return self.items[item.value]
        self.error('bad arg for getitem: %s' % item)
    def __eq__(self, other):
        return Integer(isinstance(other, List) and
                self.items == other.items, info=self)
    def __add__(self, other):
        if not isinstance(other, List):
            self.error('bad type for list.add: %s' % type(other))
        return List(self.items + other.items, info=self)
    def __contains__(self, item):
        r = 0
        for i in self:
            if (item == i).value:
                r = 1
                break
        return Integer(r, info=self)
    def len(self, ctx):
        return len(self.items)
    def __hash__(self):
        return tuple(self.items).__hash__()

@node('items')
class Dict(Node):
    def eval(self, ctx):
        return Dict({k.eval(ctx): v.eval(ctx) for k, v in
            self.items.items()}, info=self)
    def repr(self, ctx):
        return '{%s}' % ', '.join('%s: %s' % (k.repr(ctx), v.repr(ctx))
            for k, v in self.items.items())
    def __getitem__(self, item):
        if item in self.items:
            return self.items[item]
        self.error('bad arg for getitem: %s' % item)
    def __contains__(self, item):
        return Integer(item in self.items, info=self)
    def len(self, ctx):
        return len(self.items)

@node('items')
class Object(Node):
    def eval(self, ctx):
        return Object({k.eval(ctx): v.eval(ctx) for k, v
            in self.items.items()}, info=self)
    def get_attr(self, attr):
        if attr in self.items:
            return self.items[attr]
        return None
    def __eq__(self, other):
        return Integer(isinstance(other, Object) and
                self.items == other.items, info=self)

    def base_repr(self, ctx):
        return '{%s}' % ', '.join('%s:%s' % (k.repr(ctx), v.repr(ctx)) for k, v
                in self.items.items())
    def len(self, ctx):
        return self.overload(ctx, '__len__', []).value
    def str(self, ctx):
        return self.overload(ctx, '__str__', [],
                delegate=lambda: String(self.repr(ctx), info=self)).value
    def repr(self, ctx):
        return self.overload(ctx, '__repr__', [],
                delegate=lambda: String(self.base_repr(ctx), info=self)).value
    def overload(self, ctx, attr, args, delegate=None):
        # Operator overloading
        cls = self.get_attr('__class__')
        op = cls.get_attr(attr)
        if op is not None:
            return op.eval_call(ctx, [self] + args)
        if delegate:
            return delegate()
        self.error('%s unimplemented for %s' % (attr, cls.repr(ctx)), ctx=ctx)

@node('&method, &self')
class BoundMethod(Node):
    def eval(self, ctx):
        return self
    def eval_call(self, ctx, args):
        method = self.method.eval(ctx)
        args = [self.self] + args
        return method.eval_call(ctx, args)
    def repr(self, ctx):
        return '%s.%s' % (self.self.repr(ctx), self.method.repr(ctx))

@node('type, &rhs')
class UnaryOp(Node):
    def eval(self, ctx):
        rhs = self.rhs.eval(ctx)
        if self.type == 'not':
            return rhs.__not__()
        assert False
    def repr(self, ctx):
        return '(%s %s)' % (self.type.repr(ctx), self.rhs.repr(ctx))

@node('type, &lhs, &rhs')
class BinaryOp(Node):
    def eval(self, ctx):
        lhs = self.lhs.eval(ctx)
        # Check for short-circuiting bool ops. Ideally since we're purely
        # functional this doesn't matter, but builtins and other things
        # have side effects at the moment.
        if self.type in {'and', 'or'}:
            test = lhs.test_truth()
            if self.type == 'or':
                test = not test
            if test:
                return self.rhs.eval(ctx)
            return lhs

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
            '-':  'sub',
            '&':  'and',
            '|':  'or',
            '^':  'xor',
            'in': 'contains',
        }[self.type]
        operator = '__%s__' % operator

        if not hasattr(lhs, operator):
            return lhs.overload(ctx, operator, [rhs])
        return getattr(lhs, operator)(rhs)
    def repr(self, ctx):
        return '(%s %s %s)' % (self.lhs.repr(ctx), self.type, self.rhs.repr(ctx))

@node('&obj, attr')
class GetAttr(Node):
    def eval(self, ctx):
        obj = self.obj.eval(ctx)
        item = obj.get_attr(self.attr)
        if item is None:
            item = BoundMethod(GetAttr(obj.get_attr('__class__'), self.attr), obj)
        return item
    def repr(self, ctx):
        return '%s.%s' % (self.obj.repr(ctx), self.attr)

@node('&obj, &item')
class GetItem(Node):
    def eval(self, ctx):
        obj = self.obj.eval(ctx)
        item = self.item.eval(ctx)
        return obj[item]
    def repr(self, ctx):
        return '%s[%s]' % (self.obj.repr(ctx), self.item.repr(ctx))

@node('name, &rhs')
class Assignment(Node):
    def eval(self, ctx):
        value = self.rhs.eval(ctx)
        def assign_target(lhs, rhs):
            if isinstance(lhs, str):
                ctx.store(lhs, rhs)
            elif isinstance(lhs, list):
                if len(lhs) != rhs.len(ctx):
                    self.error('too %s values to unpack' %
                           ('few' if len(lhs) > len(rhs) else 'many'), ctx=ctx)
                for lhs_i, rhs_i in zip(lhs, rhs):
                    assign_target(lhs_i, rhs_i)
            else:
                assert False
        assign_target(self.name, value)
        return value
    def repr(self, ctx):
        return '%s = %s' % (self.name, self.rhs.repr(ctx))

# Exception for backing up the eval stack on return
class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value

@node('&expr')
class Return(Node):
    def eval(self, ctx):
        if self.expr:
            expr = self.expr.eval(ctx)
        else:
            expr = Nil(info=self)
        raise ReturnValue(expr)
    def repr(self, ctx):
        return 'return%s' % (' %s' % self.expr.repr(ctx) if
                self.expr else '')

@node('&expr')
class Yield(Node):
    def eval_gen(self, ctx):
        yield self.expr.eval(ctx)
    def repr(self, ctx):
        return 'yield %s' % self.expr.repr(ctx)

@node('&expr, *if_stmts, *else_stmts')
class IfElse(Node):
    def eval(self, ctx):
        expr = self.expr.eval(ctx).test_truth()
        block = self.if_stmts if expr else self.else_stmts
        value = Nil(info=self)
        for stmt in block:
            value = stmt.eval(ctx)
        return value
    def eval_gen(self, ctx):
        expr = self.expr.eval(ctx).test_truth()
        block = self.if_stmts if expr else self.else_stmts
        for stmt in block:
            yield from stmt.eval_gen(ctx)
    def repr(self, ctx):
        else_block = ''
        if self.else_stmts:
            else_block = '\nelse%s' % block_str(self.else_stmts, ctx)
        return 'if %s%s%s' % (self.expr.repr(ctx), block_str(self.if_stmts, ctx), else_block)

@node('iter, &expr, *body')
class For(Node):
    def eval(self, ctx):
        expr = self.expr.eval(ctx)
        for i in expr:
            ctx.store(self.iter, i)
            for stmt in self.body:
                stmt.eval(ctx)
        return Nil(info=self)
    def eval_gen(self, ctx):
        expr = self.expr.eval(ctx)
        for i in expr:
            ctx.store(self.iter, i)
            for stmt in self.body:
                yield from stmt.eval_gen(ctx)
    def repr(self, ctx):
        return 'for %s in %s%s' % (self.iter, self.expr.repr(ctx), block_str(self.body, ctx))

@node('&expr, *body')
class While(Node):
    def eval(self, ctx):
        while self.expr.eval(ctx).test_truth():
            for stmt in self.body:
                stmt.eval(ctx)
        return Nil(info=self)
    def eval_gen(self, ctx):
        while self.expr.eval(ctx).test_truth():
            for stmt in self.body:
                yield from stmt.eval_gen(ctx)
    def repr(self, ctx):
        return 'while %s%s' % (self.expr.repr(ctx), block_str(self.body, ctx))

@node('&fn, *args')
class Call(Node):
    def eval(self, ctx):
        fn = self.fn.eval(ctx)
        args = [a.eval(ctx) for a in self.args]
        return fn.eval_call(ctx, args)
    def repr(self, ctx):
        return '%s(%s)' % (self.fn.repr(ctx), ', '.join(s.repr(ctx) for s in self.args))

@node('ctx, name, params, *block')
class Function(Node):
    def setup(self):
        # Check if this is a generator and not a function
        self.is_generator = False
        if any(isinstance(node, Yield) for node in self.iterate_subtree()):
            if any(isinstance(node, Return) for node in self.iterate_subtree()):
                self.error('Cannot use return in a generator')
            self.is_generator = True
    def eval(self, ctx):
        return self
    def eval_call(self, ctx, args):
        ret = Nil(info=self)
        child_ctx = Context(self.name, self, self.ctx, ctx)
        for p, a in zip(self.params, args):
            child_ctx.store(p, a)
        if self.is_generator:
            return Generator(child_ctx, self.block, info=self)
        for expr in self.block:
            try:
                expr.eval(child_ctx)
            except ReturnValue as r:
                ret = r.value
                break
        return ret
    def repr(self, ctx):
        return 'def %s(%s)%s' % (self.name, ', '.join(str(s)
            for s in self.params), block_str(self.block, ctx))

@node('ctx, *block')
class Generator(Node):
    def setup(self):
        self.exhausted = False
    def eval(self, ctx):
        return self
    def __iter__(self):
        if self.exhausted:
            self.error('generator exhausted', ctx=self.ctx)
        for expr in self.block:
            yield from expr.eval_gen(self.ctx)
        self.exhausted = True

@node('name, fn')
class BuiltinFunction(Node):
    def eval_call(self, ctx, args):
        child_ctx = Context(self.name, self, None, ctx)
        return self.fn(child_ctx, args)
    def repr(self, ctx):
        return '<builtin %s>' % self.name

@node('ctx, name, *block')
class Class(Node):
    def eval(self, ctx):
        child_ctx = Context(self.name, self, self.ctx, ctx)
        for expr in self.block:
            expr.eval(child_ctx)
        items = {String(k, info=self): v.eval(ctx) for k, v
            in child_ctx.syms.items()}
        items[String('name', info=self)] = String(self.name, info=self)
        cls = Object(items, info=self)
        self.cls = cls
        ctx.store(self.name, self)
        return self
    def eval_call(self, ctx, args):
        init = self.cls.get_attr('__init__')
        if init is None:
            attrs = {}
        else:
            obj = init.eval_call(ctx, args)
            attrs = obj.items
        # Add __class__ attribute
        attrs[String('__class__', info=self)] = self
        return Object(attrs, info=self)
    def repr(self, ctx):
        return "<class '%s'>" % self.name
    def get_attr(self, attr):
        return self.cls.get_attr(attr)

@node('name, names, path, is_builtins')
class Import(Node):
    def eval(self, ctx):
        for expr in self.stmts:
            expr.eval(self.ctx)
        if self.names is None:
            obj = Object({String(k, info=self): v.eval(ctx) for k, v
                in self.ctx.syms.items()}, info=self)
            ctx.store(self.name, obj)
        else:
            for k, v in self.ctx.syms.items():
                if self.names == [] or k in self.names:
                    ctx.store(k, v.eval(ctx))
        return Nil(info=self)

    def repr(self, ctx):
        if self.names is not None:
            names = '*' if not self.names else ', '.join(self.names)
            return 'from %s import %s' % (self.name, names)
        return 'import %s' % self.name
