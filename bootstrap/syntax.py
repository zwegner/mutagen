import collections
import copy
import enum

import greenlet

import sprdpl.lex

# XXX HACK?
import sys
sys.setrecursionlimit(10000)

# This is a list, since order matters--backslashes must come first!
INV_STR_ESCAPES = [
    ['\\', '\\\\'],
    ['\n', '\\n'],
    ['\t', '\\t'],
    ['\b', '\\x08'], # HACK
]

# Info object to represent a non-existent source location for all builtins defined in Python
BUILTIN_INFO = sprdpl.lex.Info('__builtins__')

# This dictionary keeps all builtins implemented in Python. There is some rather shitty logic
# around this, where it primarily gets filled in by a decorator, but there are other modules
# that also add builtins. These will just modify this global dict when their builtins are defined,
# but they depend on this module, so we have to rely on all of these files being imported
# by a module that isn't depended on (right now, parse.py). Not pretty but it's simple.
builtins = collections.OrderedDict()

# Global variable to track the current innermost greenlet of execution for effect handling.
EFFECT_CHILD = None

# Constants to add to various classes' __hash__ implementation to distinguish their types.
# Since the hash functions are based on a tuple of their child objects, we want to make
# sure that [[0, 0]] has a different hash than {0: 0}, etc.
HASH_BASE_LIST, HASH_BASE_DICT, HASH_BASE_SET, HASH_BASE_OBJ, HASH_BASE_FN, HASH_BASE_CLASS = range(6)

# Utility functions
def get_class_name(ctx, cls):
    name = cls.get_attr(ctx, '__name__')
    if name is not None:
        return name.str(ctx)
    return type(cls).__name__

def get_type_name(ctx, obj):
    cls = obj.get_obj_class()
    if cls is not None:
        return get_class_name(ctx, cls)
    return type(obj).__name__

def check_obj_type(info, msg_type, ctx, obj, type):
    if type is not NONE:
        obj_type = obj.get_obj_class()
        if obj_type is not type:
            info.error('bad type %s for %s, expected %s' % (get_class_name(ctx, obj_type),
                msg_type, get_class_name(ctx, type)), ctx=ctx)

def analyze_scoping(ctx, stmt):
    seen = set()
    # Analyze nested functions
    for node in stmt.iterate_graph(seen, iterate_across_scopes=False):
        if isinstance(node, Scope):
            node.analyze_scoping(ctx)

class Context:
    def __init__(self, name, parent_ctx, callsite_ctx):
        self.name = name
        self.current_node = None
        self.generator_child = None
        self.parent_ctx = parent_ctx
        self.callsite_ctx = callsite_ctx
        self.syms = collections.OrderedDict()
    # Prepopulate the symbol table with all the Python-level builtin objects. See comments for
    # the builtins dictionary at the top of the file.
    def fill_in_builtins(self):
        for k, v in builtins.items():
            self.syms[k] = v.eval(self)
    def store(self, name, value):
        self.syms[name] = value
    def load(self, node, name):
        for ctx in [self, self.parent_ctx]:
            if ctx and name in ctx.syms:
                return ctx.syms[name]
        node.error('identifier %s not found' % name, ctx=self)
    def get_stack_trace(self):
        if self.callsite_ctx:
            result = self.callsite_ctx.get_stack_trace()
            if self.current_node:
                info = self.current_node.info
                result.append(' at %s in %s, line %s' % (self.name, info.filename,
                    info.lineno))
            return result
        else:
            return ['in module %s' % self.name]

class ProgramError(Exception):
    def __init__(self, stack_trace, msg):
        self.stack_trace = stack_trace
        self.msg = msg

class Node:
    def copy(self):
        if isinstance(self, Object):
            return Object(self.items.copy(), self.obj_class, info=self)
        assert isinstance(self, (List, Dict))
        return type(self)(self.items.copy(), info=self)
    def eval(self, ctx):
        return self
    def error(self, msg, ctx=None):
        stack_trace = ctx and ctx.get_stack_trace()
        msg = '%s(%i): %s' % (self.info.filename, self.info.lineno, msg)
        raise ProgramError(stack_trace, msg)
    def __ne__(self, other):
        return Boolean(not self.__eq__(other).value, info=self)
    def bool(self, ctx):
        if self.len is Node.len:
            return True
        return self.len(ctx) > 0
    def len(self, ctx):
        self.error('__len__ unimplemented for %s' % get_type_name(ctx, self), ctx=ctx)
    def str(self, ctx):
        return self.repr(ctx)
    def repr(self, ctx):
        self.error('__repr__ unimplemented for %s' % get_type_name(ctx, self), ctx=ctx)
    def get_attr(self, ctx, attr):
        if attr == '__class__':
            return self.get_obj_class()
        return None
    def get_obj_class(self):
        return None
    def get_item(self, ctx, index, info=None):
        info = info or self
        try:
            item = self[index]
        except KeyError:
            info.error('key not in dict: %s' % index.repr(ctx), ctx=ctx)
        except IndexError:
            info.error('list index out of range: %s' % index.repr(ctx), ctx=ctx)
        except TypeError:
            info.error('__getitem__ unimplemented for: %s' % get_type_name(ctx, self), ctx=ctx)
        if item is None:
            info.error('bad arg for get_item: %s' % index.repr(ctx), ctx=ctx)
        return item
    def iter(self, ctx):
        return iter(self)
    def overload(self, ctx, attr, args):
        return None
    def eval_call(self, ctx, args, kwargs):
        self.error('__call__ unimplemented for %s' % get_type_name(ctx, self), ctx=ctx)

ArgType = enum.Enum('ArgType', 'REG EDGE OPT LIST DICT')

arg_map = {'&': ArgType.EDGE, '?': ArgType.OPT, '*': ArgType.LIST, '#': ArgType.DICT}

# Weird decorator: the given arg_spec string represents a standard form for arguments
# to Node subclasses that we use to automatically create a constructor and other methods for
# the Node class (like iterate_subgraph). We use these notations:
# op       -> normal attribute (not a Node)
# &expr    -> edge attribute, used for linking to other Nodes
# ?expr    -> optional edge, either a Node or None
# *explist -> python list of Nodes
# #expdict -> python dictionary of Nodes
def node(arg_spec='', compare=False, base_type=None, ops=[]):
    args = [a.strip() for a in arg_spec.split(',') if a.strip()]
    new_args = []
    for a in args:
        if a[0] in arg_map:
            new_args.append((arg_map[a[0]], a[1:]))
        else:
            new_args.append((ArgType.REG, a))
    args = new_args

    # Decorators must return a function. This adds __init__ and some other methods
    # to a Node subclass
    def attach(node):
        def __init__(self, *iargs, info=None):
            assert len(iargs) == len(args), 'bad args, expected %s(%s)' % (node.__name__, arg_spec)

            for (arg_type, arg_name), v in zip(args, iargs):
                if arg_type == ArgType.EDGE:
                    assert isinstance(v, Node)
                elif arg_type == ArgType.OPT:
                    assert v is None or isinstance(v, Node)
                elif arg_type == ArgType.LIST:
                    assert isinstance(v, list) and all(isinstance(i, Node) for i in v)
                elif arg_type == ArgType.DICT:
                    assert isinstance(v, dict) and all(isinstance(key, Node) and
                        isinstance(value, Node) for key, value in v.items())
                setattr(self, arg_name, v)

            if info is None:
                for (arg_type, arg_name), v in zip(args, iargs):
                    if arg_type == ArgType.EDGE:
                        info = v.info
                        break
            if isinstance(info, Node):
                info = info.info
            self.info = info
            assert self.info

            if hasattr(self, 'setup'):
                self.setup()

            # XXX kinda nasty, put this here for possible use by compiler.py later
            self._uses = []

        # Iterate through all child nodes of this node
        def iterate_children(self):
            for (arg_type, arg_name) in args:
                if arg_type == ArgType.EDGE:
                    edge = getattr(self, arg_name)
                    yield edge
                elif arg_type == ArgType.OPT:
                    edge = getattr(self, arg_name)
                    if edge is not None:
                        yield edge
                elif arg_type == ArgType.LIST:
                    for edge in getattr(self, arg_name):
                        yield edge
                elif arg_type == ArgType.DICT:
                    for k, v in getattr(self, arg_name).items():
                        yield k
                        yield v

        # Iterate through the entire DAG reachable from this node, not including
        # the node. 'seen' is used to track nodes reachable through multiple paths.
        # 'iterate_across_scopes' is used to break iteration on Scope boundaries
        def iterate_subgraph(self, seen=None, iterate_across_scopes=True):
            for child in self.iterate_children():
                yield from child.iterate_graph(seen,
                        iterate_across_scopes=iterate_across_scopes)

        def iterate_graph(self, seen=None, iterate_across_scopes=True):
            seen = set() if seen is None else seen
            # XXX Ugh, we want to use the default hashing/equality for Python objects
            # for the 'seen' set, while keeping custom hashing/equality for purposes of
            # implementing dicts/etc. in the interpreter. Since it's hard to override the way
            # hashing works on a case-by-case basis, hack around it and just use the object
            # pointer as the hash key here. We don't have to worry about the GC (which could
            # in theory cause two objects to share the same address, and hence collide) since
            # 'self' is referenced here for the whole traversal of the graph below this node...
            key = id(self)
            if key not in seen:
                seen.add(key)
                yield self
                if iterate_across_scopes or not isinstance(self, Scope):
                    yield from self.iterate_subgraph(seen,
                            iterate_across_scopes=iterate_across_scopes)

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

            # Fucking Python mutable default arguments...
            all_ops = ops + ['ge', 'gt', 'le', 'lt']
        else:
            all_ops = ops

        # Generate wrappers for builtin operations without having to write
        # a shitload of boilerplate. If you're asking why we don't just
        # derive from int/str/whatever, well, we want to whitelist functionality
        # like this, and we need to box the results/check for errors.
        if all_ops:
            assert base_type
            for op in all_ops:
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
                    # isn't defined yet. Yuck.
                    result_type = Boolean if op in ['ge', 'gt', 'le', 'lt'] else node
                    return result_type(op_fn(self.value, other.value), info=self)

                setattr(node, full_op, operator)

        node.__init__ = __init__
        node.iterate_children = iterate_children
        node.iterate_graph = iterate_graph
        node.iterate_subgraph = iterate_subgraph
        node.arg_defs = args
        return node

    return attach

@node()
class None_(Node):
    def get_obj_class(self):
        return NoneClass
    def repr(self, ctx):
        return 'None'
    def __eq__(self, other):
        return Boolean(other is NONE, info=self)
    def bool(self, ctx):
        return False
    def __hash__(self):
        return 0

# Construct the None singleton, and then make sure we can't make any others
NONE = None_(info=BUILTIN_INFO)
None_.__init__ = None

# Just a proxy for an ABI function argument, used in the compiler
@node('index')
class Parameter(Node):
    def repr(self, ctx):
        return 'Param(%s)' % self.index

# Same for external symbols
@node('name')
class ExternSymbol(Node):
    def repr(self, ctx):
        return 'ExternSymbol(%s)' % self.name

@node('name')
class Identifier(Node):
    def eval(self, ctx):
        return ctx.load(self, self.name)
    def repr(self, ctx):
        return '%s' % self.name

@node('value', compare=True, base_type=str, ops=['add'])
class String(Node):
    def get_obj_class(self):
        return StrClass
    def str(self, ctx):
        return self.value
    def repr(self, ctx):
        value = self.value
        for k, v in INV_STR_ESCAPES:
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
        return None
    def __mul__(self, other):
        if not isinstance(other, Integer):
            self.error('bad type for str.mul: %s' % get_type_name(None, other))
        return String(self.value * other.value, info=self)
    def len(self, ctx):
        return len(self.value)

@node('value', compare=True, base_type=int, ops=['add', 'sub', 'mul', 'pow',
    'floordiv', 'mod', 'lshift', 'rshift', 'and', 'or', 'xor'])
class Integer(Node):
    def setup(self):
        self.value = int(self.value)
    def eval(self, ctx):
        return self
    def get_obj_class(self):
        return IntClass
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
    def get_obj_class(self):
        return BoolClass
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
    def set_item(self, ctx, item, value):
        assert isinstance(item, Integer)
        self.items[item.value] = value
    def repr(self, ctx):
        return '[%s]' % ', '.join(s.repr(ctx) for s in self.items)
    def __iter__(self):
        for I in self.items:
            yield I
    def __getitem__(self, item):
        if isinstance(item, Integer):
            return self.items[item.value]
        return None
    def get_obj_class(self):
        return ListClass
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
        return Boolean(item in self.items, info=self)
    def len(self, ctx):
        return len(self.items)
    def __hash__(self):
        return hash(tuple(self.items)) + HASH_BASE_LIST

@node('#items')
class Dict(Node):
    def eval(self, ctx):
        return Dict(collections.OrderedDict((k.eval(ctx), v.eval(ctx)) for k, v in
            self.items.items()), info=self)
    def set_item(self, ctx, item, value):
        self.items[item] = value
    def repr(self, ctx):
        return '{%s}' % ', '.join('%s: %s' % (k.repr(ctx), v.repr(ctx))
            for k, v in self.items.items())
    def __getitem__(self, item):
        return self.items[item]
    def get_obj_class(self):
        return DictClass
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
    def __sub__(self, other):
        if not isinstance(other, List):
            self.error('bad type for dict.sub: %s' % type(other))
        result = self.items.copy()
        for key in other.items:
            if key not in result:
                key.error('key not in dictionary')
            del result[key]
        return Dict(result, info=self)
    def len(self, ctx):
        return len(self.items)
    def __eq__(self, other):
        return Boolean(isinstance(other, Dict) and self.items == other.items, info=self)
    def __hash__(self):
        return hash(tuple(self.items.items())) + HASH_BASE_DICT

@node('#items')
class Set(Node):
    def eval(self, ctx):
        return Set(collections.OrderedDict((k.eval(ctx), NONE) for k, v in
            self.items.items()), info=self)
    def repr(self, ctx):
        return '{%s}' % ', '.join(k.repr(ctx) for k, v in self.items.items())
    def get_obj_class(self):
        return SetClass
    def __iter__(self):
        yield from self.items.keys()
    def __contains__(self, item):
        return Boolean(item in self.items, info=self)
    def __or__(self, other):
        if not isinstance(other, Set):
            self.error('bad type for set.add: %s' % type(other))
        result = self.items.copy()
        result.update(other.items)
        return Set(result, info=self)
    def __sub__(self, other):
        if not isinstance(other, Set):
            self.error('bad type for set.sub: %s' % type(other))
        result = self.items.copy()
        for key in other.items:
            if key not in result:
                key.error('key not in set')
            del result[key]
        return Set(result, info=self)
    def len(self, ctx):
        return len(self.items)
    def __eq__(self, other):
        return Boolean(isinstance(other, Set) and set(self.items) == set(other.items), info=self)
    def __hash__(self):
        return hash(tuple(self.items.key())) + HASH_BASE_SET

@node('#items, &obj_class')
class Object(Node):
    def eval(self, ctx):
        return Object(collections.OrderedDict((k.eval(ctx), v.eval(ctx))
            for k, v in self.items.items()), self.obj_class, info=self)
    def set_attr(self, ctx, name, value):
        if name not in self.items:
            self.error('bad attribute for modification: %s' % name, ctx=ctx)
        self.items[name] = value
    def get_attr(self, ctx, attr):
        if attr in self.items:
            return self.items[attr]
        return None
    def get_obj_class(self):
        return self.obj_class
    def base_repr(self):
        return '<%s at %#x>' % (self.obj_class.name, id(self))
    def __eq__(self, other):
        return Boolean(isinstance(other, Object) and
                self.items == other.items, info=self)
    def __hash__(self):
        return hash((hash(self.obj_class),) + tuple(self.items.items())) + HASH_BASE_OBJ
    def bool(self, ctx):
        result = (self.overload(ctx, '__bool__', []) or
                self.overload(ctx, '__len__', []))
        if not result:
            return True
        return result.value
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
    def get_item(self, ctx, item, info=None):
        return self.dispatch(ctx, '__getitem__', [item])
    def overload(self, ctx, attr, args):
        # Operator overloading
        op = self.obj_class.get_attr(ctx, attr)
        if op is not None and ctx is not None:
            return op.eval_call(ctx, [self] + args, {})
        return None
    # Equivalent to overload(), but with an error if there's no overload rather than
    # just returning None
    def dispatch(self, ctx, attr, args):
        return self.overload(ctx, attr, args) or self.error(
                '%s unimplemented for %s' % (attr, self.obj_class.repr(ctx)), ctx=ctx)

@node('&fn, *args')
class BoundFunction(Node):
    def eval_call(self, ctx, args, kwargs):
        args = self.args + list(args)
        return self.fn.eval_call(ctx, args, kwargs)
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

BINARY_OP_TABLE = {
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
    '**': 'pow',
    '//': 'floordiv',
    '%':  'mod',
    '&':  'and',
    '|':  'or',
    '^':  'xor',
    'in': 'contains',
}
BINARY_OP_TABLE = {k: '__%s__' % v for k, v in BINARY_OP_TABLE.items()}

def eval_binary_op(self, ctx, operator, lhs, rhs):
    # This is a bit of an abuse of Python operator overloading! Oh well...
    operator = BINARY_OP_TABLE[operator]

    overloaded = lhs.overload(ctx, operator, [rhs])
    if overloaded is not None:
        return overloaded
    elif not hasattr(lhs, operator):
        self.error('object of type %s cannot handle operator %s' % (
            get_type_name(ctx, lhs), operator), ctx=ctx)
    return getattr(lhs, operator)(rhs)

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
        return eval_binary_op(self, ctx, self.type, lhs, self.rhs.eval(ctx))
    def repr(self, ctx):
        lhs, rhs = (self.lhs.repr(ctx), self.rhs.repr(ctx))
        # Reverse the reversal done during parsing to turn 'x in y' -> 'y.__contains__(x)'
        if self.type == 'in':
            lhs, rhs = rhs, lhs
        return '(%s %s %s)' % (lhs, self.type, rhs)

# This function is equivalent to a user-level getattr(), where we check for the
# given attribute in the object, the class (for binding methods)
def get_attr(ctx, obj, attr, info=None, raise_errors=True):
    item = obj.get_attr(ctx, attr)
    # If the attribute doesn't exist, create a bound method with the attribute
    # from the object's class, assuming it exists.
    if item is None:
        method = obj.get_obj_class().get_attr(ctx, attr)
        if method is not None and isinstance(method, (Function, BuiltinFunction)):
            return BoundFunction(method, [obj])
        if raise_errors:
            info.error("object of type '%s' has no attribute '%s'" %
                    (get_type_name(ctx, obj), attr), ctx=ctx)
    return item

@node('&obj, attr')
class GetAttr(Node):
    def eval(self, ctx):
        obj = self.obj.eval(ctx)
        return get_attr(ctx, obj, self.attr, info=self)
    def repr(self, ctx):
        return '%s.%s' % (self.obj.repr(ctx), self.attr)

@node('&obj, &item')
class GetItem(Node):
    def eval(self, ctx):
        obj = self.obj.eval(ctx)
        index = self.item.eval(ctx)
        return obj.get_item(ctx, index, info=self)
    def repr(self, ctx):
        return '%s[%s]' % (self.obj.repr(ctx), self.item.repr(ctx))

@node('&expr')
class Assert(Node):
    def eval(self, ctx):
        if not self.expr.eval(ctx).bool(ctx):
            self.error('Assertion failed: %s' % self.expr.repr(ctx), ctx=ctx)
    def repr(self, ctx):
        return 'assert %s' % self.expr.repr(ctx)

@node('targets')
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
        for target in self.targets:
            _assign_values(ctx, target, rhs)

    def get_stores(self):
        def _get_stores(target):
            if isinstance(target, str):
                return {target}
            assert isinstance(target, list)
            stores = set()
            for t in target:
                stores |= _get_stores(t)
            return stores
        return _get_stores(self.targets)

    def repr(self, ctx):
        def target_repr(target):
            if isinstance(target, str):
                return target
            return '[%s]' % ', '.join(map(target_repr, target))
        return ' = '.join(map(target_repr, self.targets))

@node('&target, &rhs')
class Assignment(Node):
    def eval(self, ctx):
        value = self.rhs.eval(ctx)
        self.target.assign_values(ctx, value)
        return NONE
    def repr(self, ctx):
        return '%s = %s' % (self.target.repr(ctx), self.rhs.repr(ctx))

@node('&item')
class ModItem(Node):
    def eval_mod(self, ctx, expr, value):
        expr.set_item(ctx, self.item.eval(ctx), value)
    def eval_get(self, ctx, expr):
        return expr.get_item(ctx, self.item.eval(ctx), info=self)

@node('name')
class ModAttr(Node):
    def eval_mod(self, ctx, expr, value):
        expr.set_attr(ctx, self.name, value)
    def eval_get(self, ctx, expr):
        return expr.get_attr(ctx, self.name)

@node('op, *items, &value')
class ModItems(Node):
    def eval_mod(self, ctx, expr):
        def rec_eval_mod(expr, items, value):
            item, *items = items
            # If this is not the last attribute/item access in this chain, we
            # must copy the child object and recursively modify it
            if items:
                sub_item = item.eval_get(ctx, expr).copy()
                rec_eval_mod(sub_item, items, value)
                value = sub_item
            # At the innermost layer, check if we need to perform a binary op
            # on the value. XXX rather hackulant
            elif self.op != '=':
                op = self.op[:-1]
                sub_item = item.eval_get(ctx, expr)
                value = eval_binary_op(self, ctx, op, sub_item, value)
            return item.eval_mod(ctx, expr, value)
        rec_eval_mod(expr, self.items, self.value.eval(ctx))

@node('&expr, *mods')
class Modification(Node):
    def eval(self, ctx):
        expr = self.expr.eval(ctx).copy()
        for mod in self.mods:
            mod.eval_mod(ctx, expr)
        return expr

# Exception for backing up the eval stack on break/continue/return
class BreakExc(Exception):
    pass

class ContinueExc(Exception):
    pass

class ReturnExc(Exception):
    def __init__(self, value, return_node):
        self.value = value
        self.return_node = return_node

class ResumeExc(Exception):
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
        raise ReturnExc(self.expr.eval(ctx), self)
    def repr(self, ctx):
        return 'return %s' % self.expr.repr(ctx)

@node('&expr')
class Perform(Node):
    def eval(self, ctx):
        value = self.expr.eval(ctx)
        child = greenlet.getcurrent()
        return child.parent.switch(value)
    def repr(self, ctx):
        return '(perform %s)' % self.expr.repr(ctx)

@node('&expr')
class Resume(Node):
    def eval(self, ctx):
        raise ResumeExc(self.expr.eval(ctx))
    def repr(self, ctx):
        return 'resume %s' % self.expr.repr(ctx)

@node('&type, &effect_target, &block')
class EffectHandler(Node):
    def handle_effect(self, ctx, effect):
        self.effect_target.assign_values(ctx, effect)
        self.block.eval(ctx)
    def repr(self, ctx):
        return 'effect %s as %s%s' % (self.type.repr(ctx), self.effect_target.repr(ctx), self.block.repr(ctx))

@node('&type, fn')
class BuiltinEffectHandler(EffectHandler):
    def handle_effect(self, ctx, effect):
        self.fn(ctx, effect)
    def repr(self, ctx):
        return '<builtin effect handler %s>' % self.fn.__name__

@node('&block, *handlers')
class Consume(Node):
    def eval(self, ctx):
        global EFFECT_CHILD

        # Sentinel to capture the return value of the block being evaluated (in case it's an expression)
        # so we can return it normally. The greenlet API is kinda weird.
        class DumbSentinel:
            def __init__(self, value):
                self.value = value

        # Create a coroutine thread to run the consumed block.
        current = EFFECT_CHILD
        EFFECT_CHILD = greenlet.greenlet(lambda: DumbSentinel(self.block.eval(ctx)))

        # Weird control flow sentinels. See comments below
        args = []
        pass_to_parent = False
        return_value = None
        return_exc = None

        while True:
            try:
                if pass_to_parent:
                    # Weird case: if pass_to_parent is True, we need to pass an effect up the chain to our
                    # parent block (if it exists). If/when a parent resumes the effect, it will jump directly into
                    # it (EFFECT_CHILD is still set, we only need to manually pass like this in one direction).
                    # So this switch() call will only return when the child performs its next effect or returns,
                    # and we act like the return value here is the same as in the pass_to_parent=False case.
                    if current is None:
                        self.error('unhandled effect %s' % (effect.repr(ctx)), ctx=ctx)
                    effect = current.parent.switch(effect)
                else:
                    # Get the next effect from the consumed block. If it's None, the block is done.
                    effect = EFFECT_CHILD.switch(*args)
            # Be sure to catch any exceptions that happen inside the consumed block, and reraise them
            # outside of the loop below. We have to properly clean up the greenlet state (i.e. by
            # resetting EFFECT_CHILD so we don't try to pass control back to the consumed block)
            except (ReturnExc, BreakExc, ContinueExc) as e:
                return_exc = e
                break

            # Check for None as a return value from switch(). This is a sentinel value, which
            # means the block has finished.
            if isinstance(effect, DumbSentinel):
                return_value = effect.value
                break

            # Reset control flow stuff for the next iteration
            args = []
            pass_to_parent = False

            # Check if the effect can be handled by any of the handler blocks for this Consume.
            # This involves some control flow gymnastics: if we have a matching handler, we
            # have two cases: the handler resumes control flow in the enclosed block, or it
            # doesn't. And we also need to handle when no handler is found.
            # Case 1: If the handler resumes, we want to do it from the top of this block,
            #   so that we can properly capture the next effect from that switch(). And since
            #   we don't have multiply-resumable effects, we can just bubble up to here with
            #   an exception when we hit a resume (similar to a return). We save the value
            #   that we resumed with in args, to pass back into the child in the next loop iteration.
            # Case 2: If the handler doesn't resume, we break completely out of the consume block.
            #   No values need to be marshaled around, so we can just return, but we do need to
            #   reset EFFECT_CHILD so that the consumed block doesn't get restarted.
            # Case 3: No matching handler is found. We catch this in the 'else' for the for loop.
            #   Because the parent handler will resume execution directly into the child (if it
            #   resumes; if it doesn't this whole consume block won't continue executing), we
            #   want to handle the switch in the same place as the normal pass-to-child case,
            #   so we set a sentinel and continue on to the next loop iteration.
            obj_type = effect.get_obj_class()
            for handler in self.handlers:
                # Match the effect by type. This is hacky!
                # XXX we reevaluate the type expression each time, this is dumb
                type_eval = handler.type.eval(ctx)
                if obj_type is type_eval:
                    # More weirdness: reset EFFECT_CHILD to this thread while we're inside
                    # the effect handler. This means we can perform effects inside handlers,
                    # which is useful for wrapping/modifying effects. If the handler resumes,
                    # we'll restore EFFECT_CHILD to the thread running our inner block.
                    old_child = EFFECT_CHILD
                    EFFECT_CHILD = current
                    try:
                        handler.handle_effect(ctx, effect)
                    except ResumeExc as r:
                        args = [r.value]
                        EFFECT_CHILD = old_child
                        break
                    return return_value
            else:
                pass_to_parent = True

        EFFECT_CHILD.throw()
        EFFECT_CHILD = current

        if return_exc is not None:
            raise return_exc

        return return_value

    def repr(self, ctx):
        return 'consume%s %s' % (self.block.repr(ctx), ' '.join(h.repr(ctx) for h in self.handlers))

@node('&value')
class YieldedValueWrapper(Node):
    pass

@node('&expr')
class Yield(Node):
    def eval(self, ctx):
        value = YieldedValueWrapper(self.expr.eval(ctx))
        #value = self.expr.eval(ctx)
        child = greenlet.getcurrent()
        child.parent.switch(value)
    def repr(self, ctx):
        return 'yield %s' % self.expr.repr(ctx)

@node('ctx, &block')
class Generator(Node):
    def setup(self):
        self.exhausted = False
    def eval(self, ctx):
        return self
    def __iter__(self):
        if self.exhausted:
            self.error('generator exhausted', ctx=self.ctx)
        def run_generator():
            self.block.eval(self.ctx)

        global EFFECT_CHILD

        current = EFFECT_CHILD
        EFFECT_CHILD = greenlet.greenlet(run_generator)

        # Loop through the generator by passing control to the child greenlet, and either yielding values
        # when we get a YieldedValueWrapper, or passing control up to parent greenlets.
        # XXX There's too much similarity with this code and the Consume code above, try and refactor it...
        pass_to_parent = False
        while True:
            try:
                if pass_to_parent:
                    # Weird case: if pass_to_parent is True, we need to pass an effect up the chain to our
                    # parent block (if it exists). If/when a parent resumes the effect, it will jump directly into
                    # it (EFFECT_CHILD is still set, we only need to manually pass like this in one direction).
                    # So this switch() call will only return when the child performs its next effect or returns,
                    # and we act like the return value here is the same as in the pass_to_parent=False case.
                    if current is None:
                        self.error('unhandled effect %s' % (effect.repr(ctx)), ctx=ctx)
                    effect = current.parent.switch(effect)
                else:
                    # Get the next effect from the consumed block. If it's None, the block is done.
                    effect = EFFECT_CHILD.switch()
            # Sanity check: generators should not be throwing any control flow exceptions
            except (ReturnExc, BreakExc, ContinueExc) as e:
                self.error('Fatal error: got unexpected exception %s from generator' % e, ctx=ctx)

            if effect is None:
                break

            if isinstance(effect, YieldedValueWrapper):
                # Reset EFFECT_CHILD before yielding
                old_child = EFFECT_CHILD
                EFFECT_CHILD = current

                yield effect.value

                EFFECT_CHILD = old_child
                pass_to_parent = False
            else:
                pass_to_parent = True

        EFFECT_CHILD.throw()
        EFFECT_CHILD = current
        self.exhausted = True

@node('*stmts')
class Block(Node):
    def eval(self, ctx):
        for stmt in self.stmts:
            stmt.eval(ctx)
        return NONE
    def repr(self, ctx):
        block = ['%s;' % s.repr(ctx) for s in self.stmts]
        block = ['\n    '.join(s for s in b.splitlines()) for b in block]
        return ' {\n    %s\n}' % ('\n    '.join(block))

@node('&expr, &if_block, &else_block')
class IfElse(Node):
    def eval(self, ctx):
        expr = self.expr.eval(ctx).bool(ctx)
        block = self.if_block if expr else self.else_block
        return block.eval(ctx)
    def repr(self, ctx):
        else_block = ''
        if isinstance(self.else_block, IfElse):
            else_block = '\nel%s' % self.else_block.repr(ctx)
        elif isinstance(self.else_block, Block) and self.else_block.stmts:
            else_block = '\nelse%s' % self.else_block.repr(ctx)
        return 'if %s%s%s' % (self.expr.repr(ctx), self.if_block.repr(ctx), else_block)

# Conditional expressions are a weird case. We used to just subclass IfElse and print it differently,
# but needing to use blocks as expressions is hard to handle downstream. So in the parser, we desugar
# conditional expressions to a regular if-else with assignments to a temp, followed by a load of that temp.
@node('&if_else, &result')
class CondExpr(Node):
    def eval(self, ctx):
        self.if_else.eval(ctx)
        return self.result.eval(ctx)
    def repr(self, ctx):
        # Weird resugaring--should maybe sanity check here but its too annoying
        if_expr = self.if_else.if_block.stmts[0].rhs
        else_expr = self.if_else.else_block.stmts[0].rhs
        return '(%s if %s else %s)' % (if_expr.repr(ctx), self.if_else.expr.repr(ctx),
                else_expr.repr(ctx))

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
        return NONE
    def repr(self, ctx):
        return 'for %s in %s%s' % (self.target.repr(ctx), self.expr.repr(ctx),
                self.block.repr(ctx))

@node('&target, &expr, ?next')
class CompFor(Node):
    def iter_states(self, ctx):
        for values in self.expr.eval(ctx).iter(ctx):
            self.target.assign_values(ctx, values)
            if self.next:
                yield from self.next.iter_states(ctx)
            else:
                yield ctx
    def repr(self, ctx):
        return 'for %s in %s%s' % (self.target.repr(ctx), self.expr.repr(ctx),
                ' ' + self.next.repr(ctx) if self.next else '')

@node('&expr, ?next')
class CompIf(Node):
    def iter_states(self, ctx):
        if self.expr.eval(ctx).bool(ctx):
            if self.next:
                yield from self.next.iter_states(ctx)
            else:
                yield ctx
    def repr(self, ctx):
        return 'if %s%s' % (self.expr.repr(ctx), ' ' + self.next.repr(ctx) if self.next else '')

class Comprehension(Node):
    def setup(self):
        self.name = '<comprehension>'
    def specialize(self, parent_ctx, ctx):
        return self
    def get_states(self):
        yield from self.comp_for.iter_states(self.spec_ctx)

@node('&expr, &comp_for')
class ListComprehension(Comprehension):
    def eval(self, ctx):
        return List([self.expr.eval(child_ctx) for child_ctx in
            self.get_states()], info=self)
    def repr(self, ctx):
        return '[%s %s]' % (self.expr.repr(ctx), self.comp_for.repr(ctx))

@node('&key_expr, &value_expr, &comp_for')
class DictComprehension(Comprehension):
    def eval(self, ctx):
        return Dict(collections.OrderedDict(
            (self.key_expr.eval(child_ctx), self.value_expr.eval(child_ctx))
            for child_ctx in self.get_states()), info=self)
    def repr(self, ctx):
        return '{%s: %s %s}' % (self.key_expr.repr(ctx), self.value_expr.repr(ctx),
                self.comp_for.repr(ctx))

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
        return NONE
    def repr(self, ctx):
        return 'while %s%s' % (self.expr.repr(ctx), self.block.repr(ctx))

@node('&expr')
class VarArg(Node):
    def eval(self, ctx):
        return self.expr.eval(ctx)
    def repr(self, ctx):
        return '*%s' % self.expr.repr(ctx)

@node('name, &expr')
class KeywordArg(Node):
    def eval(self, ctx):
        return self.expr.eval(ctx)
    def repr(self, ctx):
        return '%s=%s' % (self.name, self.expr.repr(ctx))

@node('&expr')
class KeywordVarArg(Node):
    def eval(self, ctx):
        return self.expr.eval(ctx)
    def repr(self, ctx):
        return '**%s' % self.expr.repr(ctx)

@node('&fn, *args')
class Call(Node):
    def eval(self, ctx):
        fn = self.fn.eval(ctx)
        args = []
        kwargs = collections.OrderedDict()
        for a in self.args:
            if isinstance(a, VarArg):
                args.extend(list(a.eval(ctx).iter(ctx)))
            elif isinstance(a, KeywordArg):
                kwargs[a.name] = a.eval(ctx)
            elif isinstance(a, KeywordVarArg):
                kwdict = a.eval(ctx)
                if not isinstance(kwdict, Dict):
                    kwdict.error('keyword dictionary argument must be a '
                        'dictionary', ctx=ctx)
                for k, v in kwdict.items.items():
                    if not isinstance(k, String):
                        k.error('all keyword arguments must be strings', ctx=ctx)
                    kwargs[k.value] = v
            else:
                args.append(a.eval(ctx))
        ctx.current_node = self
        return fn.eval_call(ctx, args, kwargs)
    def repr(self, ctx):
        return '%s(%s)' % (self.fn.repr(ctx), ', '.join(s.repr(ctx) for s in self.args))

@node('name')
class VarParams(Node):
    pass

@node('name, &type, &expr')
class KeywordParam(Node):
    def get_attr(self, ctx, attr):
        if attr == 'name':
            return String(self.name, info=self)
        elif attr == 'type':
            return self.type
        elif attr == 'expr':
            return self.expr
        return None
    def eval(self, ctx):
        return KeywordParam(self.name, self.type.eval(ctx), self.expr.eval(ctx), info=self)
    def repr(self, ctx):
        if self.type is not NONE:
            return '%s: %s=%s' % (self.name, self.type.repr(ctx), self.expr.repr(ctx))
        return '%s=%s' % (self.name, self.expr.repr(ctx))

@node('name')
class KeywordVarParams(Node):
    pass

@node('params, *types, var_params, *kw_params, kw_var_params')
class Params(Node):
    def specialize(self, ctx):
        self.type_evals = [t.eval(ctx) for t in self.types]
        self.keyword_evals = {p.name: p.eval(ctx) for p in self.kw_params}
        return self
    def bind(self, obj, ctx, args, kwargs):
        # HACK: inspect context to get the call site of this function for better errors
        info = ctx.current_node
        if self.var_params:
            if len(args) < len(self.params):
                info.error('wrong number of arguments to %s, '
                    'expected at least %s, got %s' % (obj.name, len(self.params), len(args)), ctx=ctx)
            pos_args = args[:len(self.params)]
            var_args = List(args[len(self.params):], info=self)
        else:
            if len(args) != len(self.params):
                info.error('wrong number of arguments to %s, '
                    'expected %s, got %s' % (obj.name, len(self.params), len(args)), ctx=ctx)
            pos_args = args

        # Check argument types
        args = []
        for p, t, a in zip(self.params, self.type_evals, pos_args):
            check_obj_type(info, 'argument %s' % p, ctx, a, t)
            args += [(p, a)]

        # Bind var args
        if self.var_params:
            args += [(self.var_params, var_args)]

        # Bind keyword arguments
        for name, arg in kwargs.items():
            if name in self.keyword_evals:
                check_obj_type(info, 'keyword argument %s' % name, ctx, arg, self.keyword_evals[name].type)
            elif not self.kw_var_params:
                info.error('got unexpected keyword argument \'%s\'' % name, ctx=ctx)

        # Copy the kwargs, since we're going to mutate it
        kwargs = kwargs.copy()
        for kwparam in self.kw_params:
            if kwparam.name in kwargs:
                args += [(kwparam.name, kwargs[kwparam.name])]
                del kwargs[kwparam.name]
            else:
                args += [(kwparam.name, self.keyword_evals[kwparam.name].expr)]

        # Bind var args
        if self.kw_var_params:
            kwargs = collections.OrderedDict((String(k, info=self), v)
                    for k, v in kwargs.items())
            args += [(self.kw_var_params, Dict(kwargs, info=self))]

        return args
    def repr(self, ctx):
        params = []
        for p, t in zip(self.params, self.types):
            if t is not NONE:
                params.append('%s: %s' % (p, t.str(ctx)))
            else:
                params.append(p)
        if self.var_params:
            params.append('*%s' % self.var_params)
        for param in self.kw_params:
            params.append(param.str(ctx))
        if self.kw_var_params:
            params.append('**%s' % self.kw_var_params)
        return '(%s)' % ', '.join(params)
    # HACK
    def get_attr(self, ctx, attr):
        if attr == 'names':
            return List([String(param, info=self) for param in self.params], info=self)
        elif attr == 'types':
            # Note use of type_evals
            return List(self.type_evals, info=self)
        elif attr == 'kw_params':
            # Note use of keyword_evals
            return List([self.keyword_evals[k.name] for k in self.kw_params], info=self)
        elif attr in {'var_params', 'kw_var_params'}:
            value = getattr(self, attr)
            return String(value, info=self) if value is not None else NONE
        return None
    def __iter__(self):
        for I in self.params:
            yield I
        if self.var_params:
            yield self.var_params
        for p in self.kw_params:
            yield p.name
        if self.kw_var_params:
            yield self.kw_var_params

@node('&expr')
class Scope(Node):
    def eval(self, ctx):
        # When scopes are evaluated, we finally have the values of any extra
        # symbols (globals/nonlocals) that need to be passed into the scope,
        # so specialize the expression in question by duplicating it, setting
        # up a context with the current values of said symbols, and making
        # the context available to the object. Objects may also need to do
        # further specialization at this point (e.g. evaluating types of params)
        if isinstance(self.expr, Import):
            child_ctx = Context(self.expr.name, None, None)
            child_ctx.fill_in_builtins()
        else:
            child_ctx = Context(self.expr.name, ctx, None)
            for a in self.extra_args:
                child_ctx.store(a, ctx.load(self, a))
        new_expr = copy.copy(self.expr)
        new_expr.spec_ctx = child_ctx
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

        for node in self.iterate_subgraph(iterate_across_scopes=False):
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

@node('name, &params, ?return_type, &block')
class Function(Node):
    def setup(self):
        # Check if this is a generator and not a function
        self.is_generator = False
        if any(isinstance(node, Yield) for node in
                self.iterate_subgraph(iterate_across_scopes=False)):
            if any(isinstance(node, Return) for node in
                    self.iterate_subgraph(iterate_across_scopes=False)):
                self.error('Cannot use return in a generator')
            self.is_generator = True
    def specialize(self, parent_ctx, ctx):
        # Be sure to evaluate the parameter and return type expressions in
        # the context of the function declaration, and only once
        self.params = copy.copy(self.params).specialize(parent_ctx)
        if self.return_type:
            self.rt_eval = self.return_type.eval(parent_ctx)
        return self
    def eval_call(self, ctx, args, kwargs):
        child_ctx = Context(self.name, self.spec_ctx, ctx)
        for p, a in self.params.bind(self, ctx, args, kwargs):
            child_ctx.store(p, a)
        if self.is_generator:
            return Generator(child_ctx, self.block, info=self)

        try:
            self.block.eval(child_ctx)
        except ReturnExc as r:
            if self.return_type:
                check_obj_type(r.return_node, 'return value', child_ctx, r.value, self.rt_eval)
            return r.value
        return NONE
    def __eq__(self, other):
        return Boolean(self is other, info=self)
    def __hash__(self):
        return hash((self.name, self.params, self.return_type, self.block)) + HASH_BASE_FN
    def repr(self, ctx):
        ret_str = ' -> %s' % self.return_type.repr(ctx) if self.return_type else ''
        return '<function %s%s%s>' % (self.name, self.params.repr(ctx), ret_str)

@node('name, fn')
class BuiltinFunction(Node):
    def eval_call(self, ctx, args, kwargs):
        return self.fn(ctx, args, kwargs)
    def repr(self, ctx):
        return '<builtin %s>' % self.name

# XXX it seems our object model has a circular reference
# 'cls' must be None when instantiating this class. It's created when the class statement
# is evaluated. It is only an edge here so iterate_children et al will find it.
# XXX 'cls' should really be removed, and eval() should return a distinct object
@node('name, &params, &block, ?cls')
class Class(Node):
    def specialize(self, parent_ctx, ctx):
        # Be sure to evaluate the parameter type expressions in
        # the context of the function declaration, and only once
        self.params = self.params.specialize(parent_ctx)
        return self
    def eval(self, ctx):
        if not self.cls:
            child_ctx = Context(self.name, self.spec_ctx, ctx)
            self.block.eval(child_ctx)
            items = collections.OrderedDict((String(k, info=self), v)
                    for k, v in child_ctx.syms.items())
            assert '__name__' not in items and '__params__' not in items
            items[String('__name__', info=self)] = String(self.name, info=self)
            items[String('__params__', info=self)] = self.params
            self.cls = Object(items, TypeClass, info=self)
        return self
    def eval_call(self, ctx, args, kwargs):
        init = self.cls.get_attr(ctx, '__init__')
        if init is None:
            attrs = collections.OrderedDict((String(k, info=self), v)
                    for k, v in self.params.bind(self, ctx, args, kwargs))
        else:
            d = init.eval_call(ctx, args, kwargs)
            assert isinstance(d, Dict)
            attrs = d.items
        # XXX self and not self.cls?
        return Object(attrs, self, info=self)
    def repr(self, ctx):
        return "<class '%s'>" % self.name
    def get_attr(self, ctx, attr):
        return self.cls.get_attr(ctx, attr)
    def get_obj_class(self):
        return TypeClass
    def __eq__(self, other):
        return Boolean(self is other, info=self)
    def __hash__(self):
        return hash((self.name, self.params, self.block)) + HASH_BASE_CLASS

# Sentinel used for annotating builtin class method types, signifying the enclosing class
# should be substituted (the class isn't available at the time the methods are decorated,
# since they are inside the class statement). This must be a Node.
class SelfTypeSentinel(Node): pass
SELF_TYPE = SelfTypeSentinel()
SELF_TYPE.info = BUILTIN_INFO

# Superclass for every Mutagen-exposed builtin class. Mainly just some machinery for
# simplifying parameter passing and builtin methods.
class BuiltinClass(Class):
    name = None
    params = None
    def __init__(self):
        self.spec_ctx = None
        cls = type(self)
        assert cls.name is not None
        params = cls.params or Params([], [], None, [], None, info=BUILTIN_INFO)

        stmts = []
        # Nasty stuff here. We now have an instantiation of this class, so we need to
        # do some post-processing on any class attributes that needed the class.
        for attr in dir(cls):
            method = getattr(cls, attr)
            if hasattr(method, '_is_builtin_method'):
                method.name = '%s.%s' % (cls.name, method.__name__)
                # NASTY HACK! Now that the class is instantiated, fill it in in the types
                # for parameters wherever the SELF_TYPE sentinel is. This is the price
                # we pay for decent type checking without boilerplate (making sure you can't call
                # str.upper(0), etc.)
                method._params.types = [self if t == SELF_TYPE else t for t in method._params.types]
                for kwparam in method._params.kw_params:
                    if kwparam.type == SELF_TYPE:
                        kwparam.type = self
                # HACK: builtins don't get scoped/evaluated/etc. normally, so we trust that all of
                # the parameter types are already evaluated.
                method._params.type_evals = method._params.types
                method._params.keyword_evals = {p.name: p for p in method._params.kw_params}

                # Wrap the method with parameter type checking
                method = wrap_with_type_checks(method, method._params)

                stmts.append(Assignment(Target([attr], info=BUILTIN_INFO),
                    BuiltinFunction(attr, method, info=BUILTIN_INFO)))

        super().__init__(cls.name, params,
            Block(stmts, info=BUILTIN_INFO), None)

# Decorator for builtin classes, mainly just to attach a parameter specification and a name
def builtin_class(name, *, params=None, types=[], var_params=None, kw_params={}):
    def wrap(cls):
        nonlocal kw_params

        assert issubclass(cls, BuiltinClass)
        cls.name = name

        if params is not None:
            assert len(params) == len(types)
            kw_params = [KeywordParam(k, v, NONE)
                    for k, v in kw_params.items()]

            cls.params = Params(params, types, var_params, kw_params, None, info=BUILTIN_INFO)
            # HACK: builtins don't get scoped/evaluated/etc. normally, so we trust that all of
            # the parameter types are already evaluated.
            cls.params.type_evals = types
            cls.params.keyword_evals = {p.name: p for p in kw_params}

        # Kinda weird: instead of returning the actual class, return an instance
        # of the inner class. This is a Mutagen-level class, and can be used elsewhere naturally,
        # whereas the original class is mostly useless on its own, its just a convenient
        # syntactical way to group methods/params together.
        instance = cls()
        builtins[cls.name] = instance
        return instance
    return wrap

def wrap_with_type_checks(fn, params):
    def inner(ctx, args, kwargs):
        # Ignore the return value, just do the error checking
        params.bind(fn, ctx, args, kwargs)

        return fn(ctx, *args, **kwargs)
    return inner

# Decorator for builtin methods. This checks arguments, and annotates the functions so
# it can be properly exposed to user-level code.
def builtin_method(params=[], types=[], var_params=None, kw_params={}):
    kw_params = [KeywordParam(k, v, NONE)
            for k, v in kw_params.items()]
    def wrap(fn):
        nonlocal params, types
        params = ['self'] + params
        types = [SELF_TYPE] + types

        fn._params = Params(params, types, var_params, kw_params, None, info=fn)
        fn._is_builtin_method = True
        return fn
    return wrap

@builtin_class('NoneType')
class NoneClass(BuiltinClass):
    def eval_call(self, ctx, args, kwargs):
        return NONE

@builtin_class('str')
class StrClass(BuiltinClass):
    def eval_call(self, ctx, args, kwargs):
        [arg] = args
        return String(arg.str(ctx), info=arg)
    @builtin_method(['counted'], [SELF_TYPE])
    def count(ctx, self, counted):
        return Integer(self.value.count(counted.value), info=self)
    @builtin_method()
    def islower(ctx, self):
        return Boolean(self.value.islower(), info=self)
    @builtin_method()
    def lower(ctx, self):
        return String(self.value.lower(), info=self)
    @builtin_method()
    def isupper(ctx, self):
        return Boolean(self.value.isupper(), info=self)
    @builtin_method()
    def upper(ctx, self):
        return String(self.value.upper(), info=self)
    @builtin_method(['prefix'], [SELF_TYPE])
    def startswith(ctx, self, prefix):
        return Boolean(self.value.startswith(prefix.value), info=self)
    @builtin_method(['suffix'], [SELF_TYPE])
    def endswith(ctx, self, suffix):
        return Boolean(self.value.endswith(suffix.value), info=self)
    @builtin_method(['pattern', 'repl'], [SELF_TYPE, SELF_TYPE])
    def replace(ctx, self, pattern, repl):
        return String(self.value.replace(pattern.value, repl.value), info=self)
    @builtin_method(['splitter'], [SELF_TYPE])
    def split(ctx, self, splitter):
        items = [String(s, info=self) for s in self.value.split(splitter.value)]
        return List(items, info=self)
    @builtin_method(['args'], [NONE])
    def join(ctx, self, args):
        return String(self.value.join(a.value for a in args), info=self)
    @builtin_method(['encoding'], [SELF_TYPE])
    def encode(ctx, self, encoding):
        if encoding.value not in {'ascii', 'utf-8'}:
            self.error('encoding must be one of "ascii" or "utf-8"', ctx=ctx)
        try:
            encoded = self.value.encode(encoding.value)
        except UnicodeEncodeError as e:
            self.error(str(e), ctx=ctx) # just copy the Python exception message...
        # XXX create list of integers, as we don't yet have a 'bytes' object
        return List(list(Integer(i, info=self) for i in encoded), info=self)
    @builtin_method(var_params='args')
    def format(ctx, self, *args):
        args = [arg.str(ctx) for arg in args]
        return String(self.value.format(*args), info=self)

@builtin_class('int')
class IntClass(BuiltinClass):
    def eval_call(self, ctx, args, kwargs):
        if len(args) == 2:
            result = int(args[0].value, args[1].value)
        elif len(args) == 1:
            result = int(args[0].value)
        else:
            ctx.error('bad args to int()')
        return Integer(result, info=args[0])

@builtin_class('bool')
class BoolClass(BuiltinClass):
    def eval_call(self, ctx, args, kwargs):
        [arg] = args
        return Boolean(arg.bool(ctx), info=arg)

@builtin_class('list')
class ListClass(BuiltinClass):
    def eval_call(self, ctx, args, kwargs):
        [arg] = args
        return List(list(arg), info=arg)
    @builtin_method(['item'], [NONE])
    def index(ctx, self, item):
        return Integer(self.items.index(item), info=self)

@builtin_class('dict')
class DictClass(BuiltinClass):
    def eval_call(self, ctx, args, kwargs):
        [arg] = args
        return Dict(collections.OrderedDict(list(arg)), info=arg)
    @builtin_method()
    def keys(ctx, self):
        return List(list(self.items.keys()), info=self)
    @builtin_method()
    def values(ctx, self):
        return List(list(self.items.values()), info=self)

@builtin_class('set')
class SetClass(BuiltinClass):
    def eval_call(self, ctx, args, kwargs):
        items = []
        if args:
            [arg_iter] = args
            items = ((arg, NONE) for arg in arg_iter)
        return Set(collections.OrderedDict(items), info=self)
    @builtin_method()
    def pop(ctx, self):
        assert isinstance(self, Set)
        items = self.items.copy()
        if not items:
            self.error('set is empty')
        [item, _] = items.popitem()
        return List([item, Set(items, info=self)], info=self)

@builtin_class('type')
class TypeClass(BuiltinClass):
    def eval_call(self, ctx, args, kwargs):
        [arg] = args
        item = arg.get_obj_class()
        if item is None:
            self.error('object of type %s not currently part of type system' %
                    get_type_name(ctx, arg), ctx=ctx)
        return item

@builtin_class('module')
class ModuleClass(BuiltinClass):
    pass

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
            expr.eval(self.spec_ctx)
        if self.names is None:
            attrs = collections.OrderedDict((String(k, info=self), v)
                    for k, v in self.spec_ctx.syms.items())
            # Set the type of modules, and make sure we evaluate it at least once
            obj = Object(attrs, ModuleClass.eval(ctx), info=self)
            self.parent_ctx.store(self.name, obj)
        else:
            for k, v in self.spec_ctx.syms.items():
                if not self.names or k in self.names:
                    self.parent_ctx.store(k, v)
        return NONE
    def repr(self, ctx):
        stmts = '\n'.join(stmt.repr(ctx) for stmt in self.stmts)
        if self.names is not None:
            names = '*' if not self.names else ', '.join(self.names)
            return 'from %s import %s' % (self.name, names)
        return 'import %s {\n%s\n}' % (self.name, stmts)
