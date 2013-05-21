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
        return repr(self)
    def __repr__(self):
        self.error('__repr__ unimplemented for type %s' % type(self))

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
            node.__eq__ = __eq__
            node.__ne__ = __ne__
            node.__ge__ = __ge__
            node.__gt__ = __gt__
            node.__le__ = __le__
            node.__lt__ = __lt__

        node.__init__ = __init__
        node.iterate_subtree = iterate_subtree
        return node

    return attach

def block_str(block):
    block = [str(s) for s in block]
    block = ['\n    '.join(s for s in b.splitlines()) for b in block]
    return ':\n    %s' % ('\n    '.join(block))

@node()
class Nil(Node):
    def __repr__(self):
        return 'Nil'
    def __eq__(self, other):
        return Integer(isinstance(other, Nil), info=self)
    def __ne__(self, other):
        return Integer(not isinstance(other, Nil), info=self)
    def test_truth(self):
        return False

@node('name')
class Identifier(Node):
    def eval(self, ctx):
        return ctx.load(self, self.name)
    def __repr__(self):
        return '%s' % self.name

@node('value', compare=True)
class String(Node):
    def __str__(self):
        return self.value
    def __repr__(self):
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
    def __len__(self):
        return len(self.value)

@node('value', compare=True)
class Integer(Node):
    def setup(self):
        self.value = int(self.value)
    def eval(self, ctx):
        return self
    def __repr__(self):
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
    def __repr__(self):
        return '[%s]' % ', '.join(repr(s) for s in self.items)
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
    def __len__(self):
        return len(self.items)

# HACK: need a dict type
@node('*items')
class Object(Node):
    def eval(self, ctx):
        return Object([List([k.eval(ctx), v.eval(ctx)], info=self) for k, v
            in self.items], info=self)
    def get_attr(self, attr):
        results = [v for k, v in self.items if k.value == attr]
        return results[0] if results else None
    def __repr__(self):
        return '{%s}' % ', '.join('%s:%s' % (k, v) for k, v
                in self.items)
    def __eq__(self, other):
        return Integer(isinstance(other, Object) and
                self.items == other.items, info=self)

@node('&method, &self')
class BoundMethod(Node):
    def eval(self, ctx):
        return self
    def eval_call(self, ctx, args):
        method = self.method.eval(ctx)
        args = [self.self] + args
        return method.eval_call(ctx, args)
    def __repr__(self):
        return '%s.%s' % (self.self, self.method)

@node('type, &rhs')
class UnaryOp(Node):
    def eval(self, ctx):
        rhs = self.rhs.eval(ctx)
        if self.type == 'not':
            return rhs.__not__()
        assert False
    def __repr__(self):
        return '(%s %s)' % (self.type, self.rhs)

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
        }[self.type]
        operator = '__%s__' % operator
        if not hasattr(lhs, operator):
            self.error('%s unimplemented for %s' % (operator, lhs), ctx=ctx)
        return getattr(lhs, operator)(rhs)
    def __repr__(self):
        return '(%s %s %s)' % (self.lhs, self.type, self.rhs)

@node('&obj, attr')
class GetAttr(Node):
    def eval(self, ctx):
        obj = self.obj.eval(ctx)
        item = obj.get_attr(self.attr)
        if item is None:
            item = BoundMethod(GetAttr(obj.get_attr('__class__'), self.attr), obj)
        return item
    def __repr__(self):
        return '%s.%s' % (self.obj, self.attr)

@node('&obj, &item')
class GetItem(Node):
    def eval(self, ctx):
        # HACK: no real dictionaries
        obj = self.obj.eval(ctx)
        item = self.item.eval(ctx)
        return obj[item]
    def __repr__(self):
        return '%s[%s]' % (self.obj, self.item)

@node('name, &rhs')
class Assignment(Node):
    def eval(self, ctx):
        value = self.rhs.eval(ctx)
        def assign_target(lhs, rhs):
            if isinstance(lhs, str):
                ctx.store(lhs, rhs)
            elif isinstance(lhs, list):
                if len(lhs) != len(rhs):
                    self.error('too %s values to unpack' %
                           ('few' if len(lhs) > len(rhs) else 'many'), ctx=ctx)
                for lhs_i, rhs_i in zip(lhs, rhs):
                    assign_target(lhs_i, rhs_i)
            else:
                assert False
        assign_target(self.name, value)
        return value
    def __repr__(self):
        return '%s = %s' % (self.name, self.rhs)

@node('&expr, *if_stmts, *else_stmts')
class IfElse(Node):
    def eval(self, ctx):
        expr = self.expr.eval(ctx).test_truth()
        block = self.if_stmts if expr else self.else_stmts
        value = Nil(info=self)
        for stmt in block:
            value = stmt.eval(ctx)
        return value
    def __repr__(self):
        else_block = ''
        if self.else_stmts:
            else_block = '\nelse%s' % block_str(self.else_stmts)
        return 'if %s%s%s' % (self.expr, block_str(self.if_stmts), else_block)

@node('iter, &expr, *body')
class For(Node):
    def eval(self, ctx):
        expr = self.expr.eval(ctx)
        for i in expr:
            ctx.store(self.iter, i)
            for stmt in self.body:
                stmt.eval(ctx)
        return Nil(info=self)
    def __repr__(self):
        return 'for %s in %s%s' % (self.iter, self.expr, block_str(self.body))

@node('&expr, *body')
class While(Node):
    def eval(self, ctx):
        while self.expr.eval(ctx).test_truth():
            for stmt in self.body:
                stmt.eval(ctx)
        return Nil(info=self)
    def __repr__(self):
        return 'while %s%s' % (self.expr, block_str(self.body))

@node('&fn, *args')
class Call(Node):
    def eval(self, ctx):
        fn = self.fn.eval(ctx)
        args = [a.eval(ctx) for a in self.args]
        return fn.eval_call(ctx, args)
    def __repr__(self):
        return '%s(%s)' % (self.fn, ', '.join(str(s) for s in self.args))

@node('ctx, name, params, *block')
class Function(Node):
    def eval(self, ctx):
        return self
    def eval_call(self, ctx, args):
        ret = Nil(info=self)
        child_ctx = Context(self.name, self, self.ctx, ctx)
        for p, a in zip(self.params, args):
            child_ctx.store(p, a)
        for expr in self.block:
            ret = expr.eval(child_ctx)
        return ret
    def __repr__(self):
        return 'def %s(%s)%s' % (self.name, ', '.join(str(s)
            for s in self.params), block_str(self.block))

@node('name, fn')
class BuiltinFunction(Node):
    def eval_call(self, ctx, args):
        child_ctx = Context(self.name, self, None, ctx)
        return self.fn(child_ctx, args)
    def __repr__(self):
        return '<builtin %s>' % self.name

@node('ctx, name, *block')
class Class(Node):
    def eval(self, ctx):
        child_ctx = Context(self.name, self, self.ctx, ctx)
        for expr in self.block:
            ret = expr.eval(child_ctx)
        items = [List([String(k, info=self), v.eval(ctx)], info=self) for k, v
            in child_ctx.syms.items()]
        items += [List([String('name', info=self), String(self.name, info=self)], info=self)]
        cls = Object(items, info=self)
        self.cls = cls
        ctx.store(self.name, cls)
        return self
    def eval_call(self, ctx, args):
        init = self.cls.get_attr('__init__')
        if init is None:
            attrs = []
        else:
            obj = init.eval_call(ctx, args)
            attrs = obj.items
        # Add __class__ attribute
        attrs += [List([String('__class__', info=self), self], info=self)]
        return Object(attrs, info=self)
    def __repr__(self):
        return '<class %s>' % self.name
    def get_attr(self, attr):
        return self.cls.get_attr(attr)

@node('name, names, path, is_builtins')
class Import(Node):
    def eval(self, ctx):
        for expr in self.stmts:
            expr.eval(self.ctx)
        if self.names is None:
            obj = Object([List([String(k, info=self), v.eval(ctx)], info=self) for k, v
                in self.ctx.syms.items()], info=self)
            ctx.store(self.name, obj)
        else:
            for k, v in self.ctx.syms.items():
                if self.names == [] or k in self.names:
                    ctx.store(k, v.eval(ctx))
        return Nil(info=self)

    def __repr__(self):
        if self.names is not None:
            names = '*' if not self.names else ', '.join(self.names)
            return 'from %s import %s' % (self.name, names)
        return 'import %s' % self.name
