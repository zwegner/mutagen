import copy

filename = 'filename'

# This is a list, since order matters--backslashes must come first!
inv_str_escapes = [
    ['\\', '\\\\'],
    ['\n', '\\n'],
    ['\t', '\\t'],
    ['\b', '\\x08'], # HACK
]

# Utility functions
def get_class_name(ctx, cls):
    if isinstance(cls, Class):
        return cls.get_attr(ctx, '__name__').str(ctx)
    return type(cls).__name__

def get_type_name(ctx, obj):
    if isinstance(obj, Object):
        return get_class_name(ctx, obj.get_attr(ctx, '__class__'))
    return type(obj).__name__

def check_obj_type(self, msg_type, ctx, obj, type):
    obj_type = obj.get_attr(ctx, '__class__')
    if obj_type is not type:
        self.error('bad %s type %s, expected %s' % (msg_type,
            get_class_name(ctx, obj_type), get_class_name(ctx, type)), ctx=ctx)

def preprocess_program(ctx, block):
    # Analyze nested functions
    for expr in block:
        for node in expr.iterate_tree():
            if isinstance(node, Scope):
                node.analyze_scoping(ctx)

class Context:
    def __init__(self, name, parent_ctx, callsite_ctx):
        self.name = name
        self.current_node = None
        self.parent_ctx = parent_ctx
        self.callsite_ctx = callsite_ctx
        self.syms = {}
    def store(self, name, value):
        self.syms[name] = value
    def load(self, node, name):
        if name in self.syms:
            return self.syms[name]
        if self.parent_ctx and name in self.parent_ctx.syms:
            return self.parent_ctx.syms[name]
        node.error('identifier %s not found' % name, ctx=self)
    def get_stack_trace(self):
        if self.callsite_ctx:
            result = self.callsite_ctx.get_stack_trace()
        else:
            result = []
        if self.current_node:
            info = self.current_node.info
            result.append(' at %s in %s, line %s' % (self.name, info.filename,
                info.lineno))
        else:
            result.append('in module %s' % self.name)
        return result

# Info means basically filename/line number, used for reporting errors
class Info:
    def __init__(self, filename, lineno):
        self.filename = filename
        self.lineno = lineno

class ProgramError(Exception):
    def __init__(self, stack_trace, msg):
        self.stack_trace = stack_trace
        self.msg = msg

class Node:
    def eval(self, ctx):
        return self
    def eval_gen(self, ctx):
        self.eval(ctx)
        return []
    def error(self, msg, ctx=None):
        stack_trace = ctx and ctx.get_stack_trace()
        msg = '%s(%i): %s' % (self.info.filename, self.info.lineno, msg)
        raise ProgramError(stack_trace, msg)
    def __ne__(self, other):
        return Boolean(not self.__eq__(other).value, info=self)
    def bool(self, ctx):
        return self.len(ctx) > 0
    def len(self, ctx):
        self.error('__len__ unimplemented for %s' % get_type_name(ctx, self), ctx=ctx)
    def str(self, ctx):
        return self.repr(ctx)
    def repr(self, ctx):
        self.error('__repr__ unimplemented for %s' % get_type_name(ctx, self), ctx=ctx)
    def get_attr(self, ctx, attr):
        self.error('__getattr__ unimplemented for %s' % get_type_name(ctx, self), ctx=ctx)
    def iter(self, ctx):
        return iter(self)
    def overload(self, ctx, attr, args):
        return None

ARG_REG, ARG_EDGE, ARG_EDGE_LIST = list(range(3))
arg_map = {'&': ARG_EDGE, '*': ARG_EDGE_LIST}

# Weird decorator: a given arg string represents a standard form for arguments
# to Node subclasses. We use these notations:
# op, &expr, *explist
# op -> normal attribute
# &expr -> edge attribute, used for linking to other Nodes
# *explist -> python list of edges
def node(argstr='', compare=False, base_type=None, ops=[]):
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
        nonlocal ops
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

        # A generator that iterates through the AST from this node. It breaks
        # the iteration at scope barriers, since those are generally handled
        # differently whenever this is used.
        def iterate_tree(self):
            yield self
            if not isinstance(self, Scope):
                yield from self.iterate_subtree()

        # Again, but just the subtree(s) below the node.
        def iterate_subtree(self):
            for (arg_type, arg_name) in args:
                if arg_type == ARG_EDGE:
                    edge = getattr(self, arg_name)
                    yield from edge.iterate_tree()
                elif arg_type == ARG_EDGE_LIST:
                    edge_list = getattr(self, arg_name)
                    for edge in edge_list:
                        yield from edge.iterate_tree()

        # If the compare flag is set, we delegate the comparison to the
        # Python object in the 'value' attribute
        if compare:
            def __eq__(self, other):
                if isinstance(other, type(self)):
                    return Boolean(self.value == other.value, info=self)
                elif isinstance(other, type(self.value)):
                    return Boolean(self.value == other, info=self)
                return Boolean(False, info=self)
            def __hash__(self):
                return self.value.__hash__()

            node.__eq__ = __eq__
            node.__hash__ = __hash__

            # Fucking Python default arguments
            ops = ops + ['ge', 'gt', 'le', 'lt']

        # Generate wrappers for builtin operations without having to write
        # a shitload of boilerplate. If you're asking why we don't just
        # derive from int/str/whatever, well, we want to whitelist functionality
        # like this, and make it slightly more Python-agnostic
        if ops:
            assert base_type
            for op in ops:
                full_op = '__%s__' % op
                op_fn = getattr(base_type, full_op)
                # Ugh, seriously fuck lexical scoping. Better than dynamic I guess,
                # but I really just want to create a bunch of different functions
                # with different parameterizations. So we have to pass them in
                # as parameters with defaults...
                def operator(self, other, op=op, op_fn=op_fn, full_op=full_op):
                    if not isinstance(other, node):
                        self.error('bad operand type for %s.%s: %s' % (
                            base_type.__name__, op, get_type_name(None, other)))
                    # I'd like to hoist this outside of this function, but Boolean
                    # isn't defined yet.
                    result_type = Boolean if op in ['ge', 'gt', 'le', 'lt'] else node
                    return result_type(op_fn(self.value, other.value), info=self)

                setattr(node, full_op, operator)

        node.__init__ = __init__
        node.iterate_tree = iterate_tree
        node.iterate_subtree = iterate_subtree
        return node

    return attach

@node()
class None_(Node):
    def get_attr(self, ctx, attr):
        if attr == '__class__':
            return NoneClass
        return None
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

@node('value', compare=True, base_type=str, ops=['add'])
class String(Node):
    def get_attr(self, ctx, attr):
        if attr == '__class__':
            return StrClass
        return None
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
    def __mul__(self, other):
        if not isinstance(other, Integer):
            self.error('bad type for str.mul: %s' % get_type_name(None, other))
        return String(self.value * other.value, info=self)
    def len(self, ctx):
        return len(self.value)

@node('value', compare=True, base_type=int, ops=['add', 'sub', 'mul',
    'floordiv', 'mod', 'lshift', 'rshift', 'and', 'or', 'xor'])
class Integer(Node):
    def setup(self):
        self.value = int(self.value)
    def eval(self, ctx):
        return self
    def get_attr(self, ctx, attr):
        if attr == '__class__':
            return IntClass
        return None
    def repr(self, ctx):
        return '%s' % self.value
    def bool(self, ctx):
        return self.value != 0
    def __neg__(self, ctx):
        return Integer(-self.value, info=self)
    def __invert__(self, ctx):
        return Integer(~self.value, info=self)

@node('value', compare=True, base_type=bool)
class Boolean(Node):
    def setup(self):
        self.value = bool(self.value)
    def eval(self, ctx):
        return self
    def get_attr(self, ctx, attr):
        if attr == '__class__':
            return BoolClass
        return None
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
    def get_attr(self, ctx, attr):
        if attr == '__class__':
            return ListClass
        return None
    def __eq__(self, other):
        return Boolean(isinstance(other, List) and
                self.items == other.items, info=self)
    def __add__(self, other):
        if not isinstance(other, List):
            self.error('bad type for list.add: %s' % type(other))
        return List(self.items + other.items, info=self)
    def __mul__(self, other):
        if not isinstance(other, Integer):
            self.error('bad type for list.mul: %s' % type(other))
        return List(self.items * other.value, info=self)
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
        return hash(tuple(self.items))

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
        self.error('bad arg for getitem: %s' % item.str(None))
    def get_attr(self, ctx, attr):
        if attr == '__class__':
            return DictClass
        return None
    def __iter__(self):
        # XXX having key-value iteration is probably nicer than Python's
        # key iteration, but should we break compatibility? Think about this!
        for k, v in self.items.items():
            yield List([k, v], info=self)
    def __contains__(self, item):
        return Boolean(item in self.items, info=self)
    def __add__(self, other):
        if not isinstance(other, Dict):
            self.error('bad type for dict.add: %s' % type(other))
        result = self.items.copy()
        result.update(other.items)
        return Dict(result, info=self)
    def len(self, ctx):
        return len(self.items)
    def __eq__(self, other):
        return Boolean(isinstance(other, Dict) and self.items == other.items, info=self)
    def __hash__(self):
        return hash(tuple(self.items.items()))

@node('items')
class Object(Node):
    def eval(self, ctx):
        return Object({k.eval(ctx): v.eval(ctx) for k, v
            in self.items.items()}, info=self)
    def get_attr(self, ctx, attr):
        if attr in self.items:
            return self.items[attr]
        return None
    def base_repr(self):
        return '<%s at %#x>' % (self.get_attr(None, '__class__').name, id(self))
    def __eq__(self, other):
        return Boolean(isinstance(other, Object) and
                self.items == other.items, info=self)
    def __hash__(self):
        return sum(hash(k) * hash(v) for k, v in self.items.items())
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
                self.base_repr(), info=self)).value
    def iter(self, ctx):
        return self.dispatch(ctx, '__iter__', [])
    def overload(self, ctx, attr, args):
        # Operator overloading
        cls = self.get_attr(ctx, '__class__')
        op = cls.get_attr(ctx, attr)
        if op is not None and ctx is not None:
            return op.eval_call(ctx, [self] + args)
        return None
    def dispatch(self, ctx, attr, args):
        cls = self.get_attr(ctx, '__class__')
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
        elif self.type == '-':
            return rhs.__neg__(ctx)
        elif self.type == '~':
            return rhs.__invert__(ctx)
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
            '>>': 'rshift',
            '<':  'lt',
            '<=': 'le',
            '<<': 'lshift',
            '+':  'add',
            '-':  'sub',
            '*':  'mul',
            '//': 'floordiv',
            '%':  'mod',
            '&':  'and',
            '|':  'or',
            '^':  'xor',
            'in': 'contains',
        }[self.type]
        operator = '__%s__' % operator

        overloaded = lhs.overload(ctx, operator, [rhs])
        if overloaded is not None:
            return overloaded
        elif not hasattr(lhs, operator):
            self.error('object of type %s cannot handle operator %s' % (
                get_type_name(ctx, lhs), operator), ctx=ctx)
        return getattr(lhs, operator)(rhs)
    def repr(self, ctx):
        return '(%s %s %s)' % (self.lhs.repr(ctx), self.type, self.rhs.repr(ctx))

@node('&obj, attr')
class GetAttr(Node):
    def eval(self, ctx):
        obj = self.obj.eval(ctx)
        item = obj.get_attr(ctx, self.attr)
        # If the attribute doesn't exist, create a bound method with the attribute
        # from the object's class, assuming it exists.
        if item is None:
            method = obj.get_attr(ctx, '__class__').get_attr(ctx, self.attr)
            if method is None:
                self.error('object of type %s has no attribute %s' %
                        (get_type_name(ctx, obj), self.attr), ctx=ctx)
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

@node('target')
class Target(Node):
    def assign_values(self, ctx, rhs):
        def _assign_values(ctx, lhs, rhs):
            if isinstance(lhs, str):
                ctx.store(lhs, rhs)
            elif isinstance(lhs, list):
                if len(lhs) != rhs.len(ctx):
                    rhs.error('too %s values to unpack' %
                           ('few' if len(lhs) > rhs.len(ctx) else 'many'), ctx=ctx)
                for lhs_i, rhs_i in zip(lhs, rhs):
                    _assign_values(ctx, lhs_i, rhs_i)
            else:
                assert False
        return _assign_values(ctx, self.target, rhs)

    def get_stores(self):
        def _get_stores(target):
            if isinstance(target, str):
                return {target}
            assert isinstance(target, list)
            stores = set()
            for t in target:
                stores |= _get_stores(t)
            return stores
        return _get_stores(self.target)

    def repr(self, ctx):
        return str(self.target)

@node('&target, &rhs')
class Assignment(Node):
    def eval(self, ctx):
        value = self.rhs.eval(ctx)
        self.target.assign_values(ctx, value)
        return value
    def repr(self, ctx):
        return '%s = %s' % (self.target.repr(ctx), self.rhs.repr(ctx))

# Exception for backing up the eval stack on break/continue/return
class BreakExc(Exception):
    pass

class ContinueExc(Exception):
    pass

class ReturnExc(Exception):
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
        raise ReturnExc(expr)
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

@node('&target, &expr, &block')
class For(Node):
    def eval(self, ctx):
        expr = self.expr.eval(ctx)
        for i in expr.iter(ctx):
            try:
                self.target.assign_values(ctx, i)
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
                self.target.assign_values(ctx, i)
                yield from self.block.eval_gen(ctx)
            except BreakExc:
                break
            except ContinueExc:
                continue
    def repr(self, ctx):
        return 'for %s in %s%s' % (self.target.repr(ctx), self.expr.repr(ctx),
                self.block.repr(ctx))

@node('&target, &expr')
class CompIter(Node):
    def repr(self, ctx):
        return 'for %s in %s' % (self.target.repr(ctx), self.expr.repr(ctx))

class Comprehension(Node):
    def setup(self):
        self.name = '<comprehension>'
    def specialize(self, parent_ctx, ctx):
        return self
    def get_states(self):
        def iter_states(iters):
            [comp_iter, *iters] = iters
            for values in comp_iter.expr.eval(self.ctx).iter(self.ctx):
                comp_iter.target.assign_values(self.ctx, values)
                if iters:
                    yield from iter_states(iters)
                else:
                    yield self.ctx

        yield from iter_states(self.comp_iters)

@node('&expr, *comp_iters')
class ListComprehension(Comprehension):
    def eval(self, ctx):
        return List([self.expr.eval(child_ctx) for child_ctx in
            self.get_states()], info=self)
    def repr(self, ctx):
        return '[%s %s]' % (self.expr.repr(ctx),
                ' '.join(comp.repr(ctx) for comp in self.comp_iters))

@node('&key_expr, &value_expr, *comp_iters')
class DictComprehension(Comprehension):
    def eval(self, ctx):
        return Dict({self.key_expr.eval(child_ctx): self.value_expr.eval(child_ctx)
            for child_ctx in self.get_states()}, info=self)
    def repr(self, ctx):
        return '{%s: %s %s}' % (self.key_expr.repr(ctx),
                self.value_expr.repr(ctx),
                ' '.join(comp.repr(ctx) for comp in self.comp_iters))

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
        args = []
        for a in self.args:
            if isinstance(a, VarArg):
                args.extend(list(a.eval(ctx).iter(ctx)))
            else:
                args.append(a.eval(ctx))
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

# XXX types need to be edges
@node('params, star_params')
class Params(Node):
    def setup(self):
        self.params, self.types = [[p[i] for p in self.params] for i in range(2)]
    def specialize(self, ctx):
        self.type_evals = [t and t.eval(ctx) for t in self.types]
        return self
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
        for p, t, a in zip(self.params, self.type_evals, pos_args):
            if t is not None:
                check_obj_type(self, 'argument', ctx, a, t)
            args += [(p, a)]
        if self.star_params:
            args += [(self.star_params, var_args)]
        return args
    def repr(self, ctx):
        params = []
        for p, t in zip(self.params, self.type_evals):
            if t is not None:
                params.append('%s: %s' % (p, t.str(ctx)))
            else:
                params.append(p)
        if self.star_params:
            params += ['*%s' % self.star_params]
        return ', '.join(params)
    def __iter__(self):
        yield from self.params
        if self.star_params:
            yield self.star_params
    def iterate_with_types(self):
        # XXX not implemented
        assert not self.star_params
        yield from zip(self.params, self.type_evals)

@node('&expr')
class Scope(Node):
    def eval(self, ctx):
        # When scopes are evaluated, we finally have the values of any extra
        # symbols (globals/nonlocals) that need to be passed into the scope,
        # so specialize the expression in question by duplicating it, setting
        # up a context with the current values of said symbols, and making
        # the context available to the object. Objects may also need to do
        # further specialization at this point (e.g. evaluating types of params)
        child_ctx = Context(self.expr.name, None, ctx)
        if not isinstance(self.expr, Import):
            for a in self.extra_args:
                child_ctx.store(a, ctx.load(self, a))
        new_expr = copy.copy(self.expr)
        new_expr.ctx = child_ctx
        return new_expr.specialize(ctx, child_ctx).eval(ctx)
    def repr(self, ctx):
        return 'scope (%s): %s' % (self.extra_args, self.expr.repr(ctx))
    def analyze_scoping(self, ctx):
        # Keep track of all the stores/loads in this function
        # HACK
        if isinstance(self.expr, Function):
            stores = set(self.expr.params)
        else:
            stores = set()
        loads = set()

        for node in self.iterate_subtree():
            if isinstance(node, Target):
                stores.update(node.get_stores())
            elif isinstance(node, Identifier):
                loads.add(node.name)

            elif isinstance(node, Scope):
                # Find extra args by looking for variables that are read but
                # not written to in the nested function, but are written to here
                nested_globals = node.analyze_scoping(ctx)
                loads |= nested_globals

        glob = loads - stores
        self.extra_args = list(sorted(glob))

        return glob

# XXX return_type should be an edge
@node('name, &params, return_type, &block')
class Function(Node):
    def setup(self):
        # Check if this is a generator and not a function
        self.is_generator = False
        if any(isinstance(node, Yield) for node in self.iterate_subtree()):
            if any(isinstance(node, Return) for node in self.iterate_subtree()):
                self.error('Cannot use return in a generator')
            self.is_generator = True
    def specialize(self, parent_ctx, ctx):
        # Be sure to evaluate the parameter and return type expressions in
        # the context of the function declaration, and only once
        self.params = copy.copy(self.params).specialize(parent_ctx)
        if self.return_type:
            self.rt_eval = self.return_type.eval(parent_ctx)
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
        except ReturnExc as r:
            ret = r.value
            if self.return_type:
                check_obj_type(self, 'return value', ctx, ret, self.rt_eval)

        return ret
    def repr(self, ctx):
        ret_str = ' -> %s' % self.return_type.repr(ctx) if self.return_type else ''
        return 'def %s(%s)%s%s' % (self.name, self.params.repr(ctx),
                ret_str, self.block.repr(ctx))

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
        return self.fn(self, ctx, args)
    def repr(self, ctx):
        return '<builtin %s>' % self.name

@node('name, &params, &block')
class Class(Node):
    def specialize(self, parent_ctx, ctx):
        # Be sure to evaluate the parameter type expressions in
        # the context of the function declaration, and only once
        self.params = self.params.specialize(parent_ctx)
        return self
    def eval(self, ctx):
        if not hasattr(self, 'cls'):
            child_ctx = Context(self.name, self.ctx, ctx)
            self.block.eval(child_ctx)
            items = {String(k, info=self): v.eval(ctx) for k, v
                in child_ctx.syms.items()}
            items[String('__name__', info=self)] = String(self.name, info=self)
            items[String('__class__', info=self)] = TypeClass
            self.cls = Object(items, info=self)
        return self
    def eval_call(self, ctx, args):
        init = self.cls.get_attr(ctx, '__init__')
        if init is None:
            attrs = {String(k, info=self): v.eval(ctx) for k, v in
                    self.params.bind(self, ctx, args)}
        else:
            ctx.current_node = self
            d = init.eval_call(ctx, args)
            assert isinstance(d, Dict)
            attrs = d.items
        # Add __class__ attribute
        attrs[String('__class__', info=self)] = self
        return Object(attrs, info=self)
    def repr(self, ctx):
        return "<class '%s'>" % self.name
    def get_attr(self, ctx, attr):
        return self.cls.get_attr(ctx, attr)
    def __eq__(self, other):
        return Boolean(self is other, info=self)
    def __hash__(self):
        return hash((self.name, self.params, self.block))

builtin_info = Info('__builtins__', 0)

class BuiltinClass(Class):
    def __init__(self, name):
        super().__init__(name, Params([], None, info=builtin_info), None)
        self.ctx = None
        methods = set(dir(type(self))) - set(dir(BuiltinClass))
        stmts = []
        for name in methods:
            fn = getattr(self.__class__, name)
            stmts.append(Assignment(Target(name, info=self),
                BuiltinFunction(name, fn, info=builtin_info)))
        self.block = Block(stmts, info=builtin_info)

class BuiltinNone(BuiltinClass):
    def eval_call(self, ctx, args):
        self.error('NoneType is not callable')

NoneClass = BuiltinNone('NoneType')

class BuiltinStr(BuiltinClass):
    def eval_call(self, ctx, args):
        [arg] = args
        return String(arg.str(ctx), info=arg)
    def islower(obj, ctx, args):
        [arg] = args
        return Boolean(arg.value.islower(), info=arg)
    def lower(obj, ctx, args):
        [arg] = args
        return String(arg.value.lower(), info=arg)
    def isupper(obj, ctx, args):
        [arg] = args
        return Boolean(arg.value.isupper(), info=arg)
    def upper(obj, ctx, args):
        [arg] = args
        return String(arg.value.upper(), info=arg)
    def startswith(obj, ctx, args):
        [arg, suffix] = args
        return Boolean(arg.value.startswith(suffix.value), info=arg)
    def endswith(obj, ctx, args):
        [arg, suffix] = args
        return Boolean(arg.value.endswith(suffix.value), info=arg)
    def replace(obj, ctx, args):
        [arg, pattern, repl] = args
        return String(arg.value.replace(pattern.value, repl.value), info=arg)
    def join(obj, ctx, args):
        [sep, args] = args
        return String(sep.value.join(a.value for a in args), info=sep)
    def encode(obj, ctx, args):
        [arg, encoding] = args
        if not isinstance(encoding, String) or encoding.value not in {'ascii',
                'utf-8'}:
            obj.error('encoding must be one of "ascii" or "utf-8"', ctx=ctx)
        try:
            encoded = arg.value.encode(encoding.value)
        except UnicodeEncodeError as e:
            obj.error(str(e), ctx=ctx) # just copy the Python exception message...
        # XXX create list of integers, as we don't yet have a 'bytes' object
        return List(list(Integer(i, info=arg) for i in encoded), info=arg)
    def format(obj, ctx, args):
        [fmt, *args] = args
        if not isinstance(fmt, String):
            obj.error('encoding must be one of "ascii" or "utf-8"', ctx=ctx)
        args = [arg.str(ctx) for arg in args]
        return String(fmt.value.format(*args), info=fmt)

StrClass = BuiltinStr('str')

class BuiltinInt(BuiltinClass):
    def eval_call(self, ctx, args):
        if len(args) == 2:
            result = int(args[0].value, args[1].value)
        elif len(args) == 1:
            result = int(args[0].value)
        else:
            ctx.error('bad args to int()')
        return Integer(result, info=args[0])

IntClass = BuiltinInt('int')

class BuiltinBool(BuiltinClass):
    def eval_call(self, ctx, args):
        [arg] = args
        return Boolean(arg.bool(ctx), info=arg)

BoolClass = BuiltinBool('bool')

class BuiltinList(BuiltinClass):
    def eval_call(self, ctx, args):
        [arg] = args
        return List(list(arg), info=arg)

ListClass = BuiltinList('list')

class BuiltinDict(BuiltinClass):
    def eval_call(self, ctx, args):
        [arg] = args
        return Dict(dict(arg), info=arg)
    def keys(obj, ctx, args):
        [arg] = args
        return List(list(arg.items.keys()), info=arg)
    def values(obj, ctx, args):
        [arg] = args
        return List(list(arg.items.values()), info=arg)

DictClass = BuiltinDict('dict')

class BuiltinType(BuiltinClass):
    def eval_call(self, ctx, args):
        [arg] = args
        item = arg.get_attr(ctx, '__class__')
        if item is None:
            self.error('object of type %s not currently part of type system' %
                    get_type_name(ctx, arg), ctx=ctx)
        return item

TypeClass = BuiltinType('type')

@node('*stmts, name, names, path, is_builtins')
class Import(Node):
    def specialize(self, parent_ctx, ctx):
        # XXX Store off the parent context. When imports are evaluated, we
        # have to fill in the imported value in the importing context. And
        # because of "from module import *", we can't determine at parse time
        # what symbols actually get set! So we cheat and just write directly
        # into the context from which we're being imported.
        self.parent_ctx = parent_ctx
        return self
    def eval(self, ctx):
        for expr in self.stmts:
            expr.eval(self.ctx)
        if self.names is None:
            obj = Object({String(k, info=self): v for k, v
                in self.ctx.syms.items()}, info=self)
            self.parent_ctx.store(self.name, obj)
        else:
            for k, v in self.ctx.syms.items():
                if not self.names or k in self.names:
                    self.parent_ctx.store(k, v)
        return None_(info=self)
    def repr(self, ctx):
        stmts = '\n'.join(stmt.repr(ctx) for stmt in self.stmts)
        if self.names is not None:
            names = '*' if not self.names else ', '.join(self.names)
            return 'from %s import %s' % (self.name, names)
        return 'import %s {\n%s\n}' % (self.name, stmts)
