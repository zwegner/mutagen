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
    def __init__(self, name, global_ctx, parent):
        self.name = name
        self.current_node = None
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
    # XXX Should this be here? What should it be named?
    def initialize(self, block):
        # Analyze nested functions
        self.lifted_lambdas = []
        for expr in block:
            expr.lift_lambdas(self)
        return self.lifted_lambdas + block
    def print_stack(self):
        if self.parent:
            self.parent.print_stack()
        if self.current_node:
            info = self.current_node.info
            print(' at %s in %s, line %s' % (self.name, info.filename,
                info.lineno), file=sys.stderr)
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
    def __ne__(self, other):
        return Boolean(not self.__eq__(other).value, info=self)
    def bool(self, ctx):
        return self.len(ctx) > 0
    def len(self, ctx):
        self.error('__len__ unimplemented for %s' % type(self), ctx=ctx)
    def str(self, ctx):
        return self.repr(ctx)
    def repr(self, ctx):
        self.error('__repr__ unimplemented for %s' % type(self), ctx=ctx)
    def iter(self, ctx):
        return iter(self)
    def overload(self, ctx, attr, args):
        return None
    # Lift any nested functions
    def lift_lambdas(self, ctx):
        # Keep track of all the stores/loads in this function
        # HACK
        if isinstance(self, Function):
            stores = set(self.params)
        else:
            stores = set()
        loads = set()

        # Python coroutines are such a drag to use man. send() also yields
        # another value, so the control flow is crazy. At least it's possible...
        subtree_gen = iter(self.iterate_subtree())
        returned_node = None
        while True:
            try:
                if returned_node:
                    node = returned_node
                    returned_node = None
                else:
                    node = next(subtree_gen)
            except StopIteration:
                break
            if isinstance(node, Assignment):
                stores.add(node.name)
            elif isinstance(node, Identifier):
                loads.add(node.name)
            elif isinstance(node, Function):
                # Find extra args by looking for variables that are read but
                # not written to in the nested function, but are written to here
                nested_locals, nested_globals = node.lift_lambdas(ctx)
                loads |= nested_globals

                extra_args = list(sorted(nested_globals))
                if extra_args:
                    node.params.add_extra_args(extra_args)

                    ctx.lifted_lambdas.append(node)
                    try:
                        returned_node = subtree_gen.send(
                                LiftedLambda(node, extra_args))
                    except StopIteration:
                        break

        return stores, loads - stores

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

        # A generator that iterates through the AST below this node.
        # It breaks the iteration at nested function barriers, since
        # those represent a new scope and are handled specially when
        # this generator is used. This generator also accepts values
        # as a coroutine, which are used as replacements in the AST.
        # This allows us to do easy AST rewriting, but relies on the
        # AST actually being a tree.
        def iterate_subtree(self, include_self=False):
            if include_self:
                replacement = yield self
                if replacement is not None:
                    return replacement
                # HACK?
                if isinstance(self, Function):
                    return
            for (arg_type, arg_name) in args:
                if arg_type == ARG_EDGE:
                    edge = getattr(self, arg_name)
                    replacement = yield from edge.iterate_subtree(include_self=True)
                    if replacement is not None:
                        setattr(self, arg_name, replacement)
                elif arg_type == ARG_EDGE_LIST:
                    edge_list = getattr(self, arg_name)
                    for i, edge in enumerate(edge_list):
                        replacement = yield from edge.iterate_subtree(include_self=True)
                        if replacement is not None:
                            # XXX assumes list
                            edge_list[i] = replacement

        # If the compare flag is set, we delegate the comparison to the
        # Python object in the 'value' attribute
        if compare:
            def __eq__(self, other):
                if isinstance(other, type(self)):
                    return Boolean(self.value == other.value, info=self)
                elif isinstance(other, type(self.value)):
                    return Boolean(self.value == other, info=self)
                return Boolean(False, info=self)
            def __ge__(self, other):
                if not isinstance(other, type(self)):
                    self.error('uncomparable types: %s, %s' % (type(self), type(other)))
                return Boolean(self.value >= other.value, info=self)
            def __gt__(self, other):
                if not isinstance(other, type(self)):
                    self.error('uncomparable types: %s, %s' % (type(self), type(other)))
                return Boolean(self.value > other.value, info=self)
            def __le__(self, other):
                if not isinstance(other, type(self)):
                    self.error('uncomparable types: %s, %s' % (type(self), type(other)))
                return Boolean(self.value <= other.value, info=self)
            def __lt__(self, other):
                if not isinstance(other, type(self)):
                    self.error('uncomparable types: %s, %s' % (type(self), type(other)))
                return Boolean(self.value < other.value, info=self)
            def __hash__(self):
                return self.value.__hash__()
            node.__eq__ = __eq__
            node.__ge__ = __ge__
            node.__gt__ = __gt__
            node.__le__ = __le__
            node.__lt__ = __lt__
            node.__hash__ = __hash__

        node.__init__ = __init__
        node.iterate_subtree = iterate_subtree
        return node

    return attach

@node()
class None_(Node):
    def repr(self, ctx):
        return 'None'
    def __eq__(self, other):
        return Boolean(isinstance(other, None_), info=self)
    def bool(self, ctx):
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
    def bool(self, ctx):
        return self.value != 0
    def __add__(self, other):
        if not isinstance(other, Integer):
            self.error('bad type for int.add: %s' % type(other))
        return Integer(self.value + other.value, info=self)
    def __sub__(self, other):
        if not isinstance(other, Integer):
            self.error('bad type for int.sub: %s' % type(other))
        return Integer(self.value - other.value, info=self)
    def __mul__(self, other):
        if not isinstance(other, Integer):
            self.error('bad type for int.sub: %s' % type(other))
        return Integer(self.value * other.value, info=self)

@node('value', compare=True)
class Boolean(Node):
    def setup(self):
        self.value = bool(self.value)
    def eval(self, ctx):
        return self
    def repr(self, ctx):
        return '%s' % self.value
    def bool(self, ctx):
        return self.value
    # HACK: this affects equality, hope this doesn't screw anything else up...
    def __bool__(self):
        return self.value

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
            try:
                return self.items[item.value]
            except IndexError:
                self.error('list index out of range: %s' % item.value)
        self.error('bad arg for getitem: %s' % item)
    def __eq__(self, other):
        return Boolean(isinstance(other, List) and
                self.items == other.items, info=self)
    def __add__(self, other):
        if not isinstance(other, List):
            self.error('bad type for list.add: %s' % type(other))
        return List(self.items + other.items, info=self)
    def __contains__(self, item):
        r = False
        for i in self:
            if (item == i).value:
                r = True
                break
        return Boolean(r, info=self)
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
        return Boolean(item in self.items, info=self)
    def len(self, ctx):
        return len(self.items)
    def __eq__(self, other):
        return Boolean(isinstance(other, Dict) and self.items == other.items, info=self)
    def __hash__(self):
        return tuple(self.items.items()).__hash__()

@node('items')
class Object(Node):
    def eval(self, ctx):
        return Object({k.eval(ctx): v.eval(ctx) for k, v
            in self.items.items()}, info=self)
    def get_attr(self, attr):
        if attr in self.items:
            return self.items[attr]
        return None
    def base_repr(self, ctx):
        return '{%s}' % ', '.join('%s:%s' % (k.repr(ctx), v.repr(ctx)) for k, v
                in self.items.items())
    def __eq__(self, other):
        return Boolean(isinstance(other, Object) and
                self.items == other.items, info=self)
    def bool(self, ctx):
        return (self.overload(ctx, '__bool__', []) or
                self.dispatch(ctx, '__len__', [])).value
    def len(self, ctx):
        return self.dispatch(ctx, '__len__', []).value
    def str(self, ctx):
        return (self.overload(ctx, '__str__', []) or String(
                self.repr(ctx), info=self)).value
    def repr(self, ctx):
        return (self.overload(ctx, '__repr__', []) or String(
                self.base_repr(ctx), info=self)).value
    def iter(self, ctx):
        return self.dispatch(ctx, '__iter__', [])
    def overload(self, ctx, attr, args):
        # Operator overloading
        cls = self.get_attr('__class__')
        op = cls.get_attr(attr)
        if op is not None:
            return op.eval_call(ctx, [self] + args)
        return None
    def dispatch(self, ctx, attr, args):
        cls = self.get_attr('__class__')
        return self.overload(ctx, attr, args) or self.error(
                '%s unimplemented for %s' % (attr, cls.repr(ctx)), ctx=ctx)

@node('&fn, *args')
class BoundFunction(Node):
    def eval_call(self, ctx, args):
        args = self.args + args
        return self.fn.eval_call(ctx, args)
    def repr(self, ctx):
        return '<bound-fn %s(%s, ...)>' % (self.fn.repr(ctx),
                ', '.join(a.repr(ctx) for a in self.args))

@node('type, &rhs')
class UnaryOp(Node):
    def eval(self, ctx):
        rhs = self.rhs.eval(ctx)
        if self.type == 'not':
            return Boolean(not rhs.bool(ctx), info=self)
        assert False
    def repr(self, ctx):
        return '(%s %s)' % (self.type, self.rhs.repr(ctx))

@node('type, &lhs, &rhs')
class BinaryOp(Node):
    def eval(self, ctx):
        lhs = self.lhs.eval(ctx)
        # Check for short-circuiting bool ops. Ideally since we're purely
        # functional this doesn't matter, but builtins and other things
        # have side effects at the moment.
        if self.type in {'and', 'or'}:
            test = lhs.bool(ctx)
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
            '*':  'mul',
            '&':  'and',
            '|':  'or',
            '^':  'xor',
            'in': 'contains',
        }[self.type]
        operator = '__%s__' % operator

        return lhs.overload(ctx, operator, [rhs]) or getattr(lhs,
                operator)(rhs)
    def repr(self, ctx):
        return '(%s %s %s)' % (self.lhs.repr(ctx), self.type, self.rhs.repr(ctx))

@node('&obj, attr')
class GetAttr(Node):
    def eval(self, ctx):
        obj = self.obj.eval(ctx)
        item = obj.get_attr(self.attr)
        if item is None:
            method = GetAttr(obj.get_attr('__class__'), self.attr).eval(ctx)
            item = BoundFunction(method, [obj])
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

@node('&expr')
class Assert(Node):
    def eval(self, ctx):
        if not self.expr.eval(ctx).bool(ctx):
            self.error('Assertion failed: %s' % self.expr.repr(ctx), ctx=ctx)
    def repr(self, ctx):
        return 'assert %s' % self.expr.repr(ctx)

def assign_target(ctx, lhs, rhs):
    if isinstance(lhs, str):
        ctx.store(lhs, rhs)
    elif isinstance(lhs, list):
        if len(lhs) != rhs.len(ctx):
            self.error('too %s values to unpack' %
                   ('few' if len(lhs) > len(rhs) else 'many'), ctx=ctx)
        for lhs_i, rhs_i in zip(lhs, rhs):
            assign_target(ctx, lhs_i, rhs_i)
    else:
        assert False

@node('name, &rhs')
class Assignment(Node):
    def eval(self, ctx):
        value = self.rhs.eval(ctx)
        assign_target(ctx, self.name, value)
        return value
    def repr(self, ctx):
        return '%s = %s' % (self.name, self.rhs.repr(ctx))

# Exception for backing up the eval stack on break/continue/return
class BreakExc(Exception):
    pass

class ContinueExc(Exception):
    pass

class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value

@node()
class Break(Node):
    def eval(self, ctx):
        raise BreakExc()
    def repr(self, ctx):
        return 'break'

@node()
class Continue(Node):
    def eval(self, ctx):
        raise ContinueExc()
    def repr(self, ctx):
        return 'continue'

@node('&expr')
class Return(Node):
    def eval(self, ctx):
        if self.expr:
            expr = self.expr.eval(ctx)
        else:
            expr = None_(info=self)
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

@node('*stmts')
class Block(Node):
    def eval(self, ctx):
        for stmt in self.stmts:
            stmt.eval(ctx)
        return None_(info=self)
    def eval_gen(self, ctx):
        for stmt in self.stmts:
            yield from stmt.eval_gen(ctx)
    def repr(self, ctx):
        block = [s.repr(ctx) for s in self.stmts]
        block = ['\n    '.join(s for s in b.splitlines()) for b in block]
        return ':\n    %s' % ('\n    '.join(block))

@node('&expr, &if_block, &else_block')
class IfElse(Node):
    def eval(self, ctx):
        expr = self.expr.eval(ctx).bool(ctx)
        block = self.if_block if expr else self.else_block
        return block.eval(ctx)
    def eval_gen(self, ctx):
        expr = self.expr.eval(ctx).bool(ctx)
        block = self.if_block if expr else self.else_block
        yield from block.eval_gen(ctx)
    def repr(self, ctx):
        else_block = ''
        if self.else_block:
            else_block = '\nelse%s' % self.else_block.repr(ctx)
        return 'if %s%s%s' % (self.expr.repr(ctx), self.if_block.repr(ctx), else_block)

@node('iter, &expr, &block')
class For(Node):
    def eval(self, ctx):
        expr = self.expr.eval(ctx)
        for i in expr.iter(ctx):
            try:
                assign_target(ctx, self.iter, i)
                self.block.eval(ctx)
            except BreakExc:
                break
            except ContinueExc:
                continue
        return None_(info=self)
    def eval_gen(self, ctx):
        expr = self.expr.eval(ctx)
        for i in expr.iter(ctx):
            try:
                ctx.store(self.iter, i)
                yield from self.block.eval_gen(ctx)
            except BreakExc:
                break
            except ContinueExc:
                continue
    def repr(self, ctx):
        return 'for %s in %s%s' % (self.iter, self.expr.repr(ctx),
                self.block.repr(ctx))

@node('&expr, &block')
class While(Node):
    def eval(self, ctx):
        while self.expr.eval(ctx).bool(ctx):
            try:
                self.block.eval(ctx)
            except BreakExc:
                break
            except ContinueExc:
                continue
        return None_(info=self)
    def eval_gen(self, ctx):
        while self.expr.eval(ctx).bool(ctx):
            try:
                yield from self.block.eval_gen(ctx)
            except BreakExc:
                break
            except ContinueExc:
                continue
    def repr(self, ctx):
        return 'while %s%s' % (self.expr.repr(ctx), self.block.repr(ctx))

@node('&fn, *args')
class Call(Node):
    def eval(self, ctx):
        fn = self.fn.eval(ctx)
        args = [a.eval(ctx) for a in self.args]
        ctx.current_node = self
        return fn.eval_call(ctx, args)
    def repr(self, ctx):
        return '%s(%s)' % (self.fn.repr(ctx), ', '.join(s.repr(ctx) for s in self.args))

@node('&expr')
class VarArg(Node):
    def eval(self, ctx):
        return self.expr.eval(ctx)
    def repr(self, ctx):
        return '*%s' % self.expr.repr(ctx)

@node('&fn, *args')
class CallVarArgs(Node):
    def eval(self, ctx):
        fn = self.fn.eval(ctx)
        args = []
        for a in self.args:
            if isinstance(a, VarArg):
                args.extend(list(a.eval(ctx).iter(ctx)))
            else:
                args.append(a.eval(ctx))
        return fn.eval_call(ctx, args)
    def repr(self, ctx):
        return '%s(%s)' % (self.fn.repr(ctx), ', '.join(s.repr(ctx) for s in self.args))

@node('params, star_params')
class Params(Node):
    def setup(self):
        self.params, self.types = [[p[i] for p in self.params] for i in range(2)]
    def bind(self, obj, ctx, args):
        if self.star_params:
            if len(args) < len(self.params):
                self.error('wrong number of arguments to %s, '
                'expected at least %s' % (obj.name, len(self.params)), ctx=ctx)
            pos_args = args[:len(self.params)]
            var_args = List(args[len(self.params):], info=self)
        else:
            if len(args) != len(self.params):
                self.error('wrong number of arguments to %s, '
                'expected %s' % (obj.name, len(self.params)), ctx=ctx)
            pos_args = args

        # Check argument types
        args = []
        for p, t, a in zip(self.params, self.types, pos_args):
            if t is not None:
                # HACK
                if not isinstance(a, Object):
                    self.error('passing non-object argument of type %s to ' \
                            'parameter of type %s' % (type(a).__name__,
                                t.repr(ctx)), ctx=ctx)
                arg_type = a.get_attr('__class__')
                t = t.eval(ctx)
                if arg_type is not t:
                    self.error('bad argument type %s, expected %s' % (
                        arg_type.repr(ctx), t.repr(ctx)), ctx=ctx)
            args += [(p, a)]
        if self.star_params:
            args += [(self.star_params, var_args)]
        return args
    def add_extra_args(self, args):
        self.params = args + self.params
        self.types = [None] * len(args) + self.types
    def repr(self, ctx):
        params = self.params
        if self.star_params:
            params += ['*%s' % self.star_params]
        return ', '.join(params)
    def __iter__(self):
        yield from self.params
        if self.star_params:
            yield self.star_params

@node('&fn, *extra_args')
class LiftedLambda(Node):
    def eval(self, ctx):
        # Load the args once when the lambda is evaluated first (not called)
        arg_values = [ctx.load(self, a) for a in self.extra_args]
        return BoundFunction(self.fn, arg_values)
    def repr(self, ctx):
        return '<lifted-lambda %s(%s)>' % (self.fn.name, ', '.join(s for s in
            self.extra_args))

@node('ctx, name, &params, &block')
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
        child_ctx = Context(self.name, self.ctx, ctx)
        for p, a in self.params.bind(self, ctx, args):
            child_ctx.store(p, a)
        if self.is_generator:
            return Generator(child_ctx, self.block, info=self)

        ret = None_(info=self)
        try:
            self.block.eval(child_ctx)
        except ReturnValue as r:
            ret = r.value
        return ret
    def repr(self, ctx):
        return 'def %s(%s)%s' % (self.name, self.params.repr(ctx),
                self.block.repr(ctx))

@node('ctx, &block')
class Generator(Node):
    def setup(self):
        self.exhausted = False
    def eval(self, ctx):
        return self
    def __iter__(self):
        if self.exhausted:
            self.error('generator exhausted', ctx=self.ctx)
        yield from self.block.eval_gen(self.ctx)
        self.exhausted = True

@node('name, fn')
class BuiltinFunction(Node):
    def eval_call(self, ctx, args):
        child_ctx = Context(self.name, None, ctx)
        return self.fn(self, child_ctx, args)
    def repr(self, ctx):
        return '<builtin %s>' % self.name

@node('ctx, name, &params, &block')
class Class(Node):
    def eval(self, ctx):
        child_ctx = Context(self.name, self.ctx, ctx)
        self.block.eval(child_ctx)
        items = {String(k, info=self): v.eval(ctx) for k, v
            in child_ctx.syms.items()}
        items[String('name', info=self)] = String(self.name, info=self)
        items[String('__class__', info=self)] = Type
        self.cls = Object(items, info=self)
        return self
    def eval_call(self, ctx, args):
        init = self.cls.get_attr('__init__')
        if init is None:
            attrs = {String(k, info=self): v.eval(ctx) for k, v in
                    self.params.bind(self, ctx, args)}
        else:
            d = init.eval_call(ctx, args)
            assert isinstance(d, Dict)
            attrs = d.items
        # Add __class__ attribute
        attrs[String('__class__', info=self)] = self
        return Object(attrs, info=self)
    def repr(self, ctx):
        return "<class '%s'>" % self.name
    def get_attr(self, attr):
        return self.cls.get_attr(attr)
    def __eq__(self, other):
        return Boolean(self is other, info=self)

builtin_info = Info('__builtins__', 0)
Type = Class(None, 'Type', Params([], None, info=builtin_info),
        Block([], info=builtin_info), info=builtin_info)

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
        return None_(info=self)
    def repr(self, ctx):
        if self.names is not None:
            names = '*' if not self.names else ', '.join(self.names)
            return 'from %s import %s' % (self.name, names)
        return 'import %s' % self.name
