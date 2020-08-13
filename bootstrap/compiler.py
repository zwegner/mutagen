#!/usr/bin/env python3
import collections.abc
import os
import struct
import sys

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)

__package__ = 'bootstrap'

from .sprdpl import parse as libparse

from . import asm
from . import elf
from . import lir
from . import parse
from . import regalloc
from . import syntax
from .syntax import ArgType

# Dumb alias
BI = syntax.BUILTIN_INFO

def add_to(cls):
    def deco(fn):
        setattr(cls, fn.__name__, fn)
        return fn
    return deco

def gen_assign(name, expr, info):
    assign = syntax.Assignment(syntax.Target([name], info=info), expr, info=info)
    add_node_usages(assign)
    return assign

# Some extra syntax node types, basically wrappers for LIR type stuff

@syntax.node('name')
class ExternSymbol(syntax.Node):
    def repr(self, ctx):
        return 'ExternSymbol(%s)' % self.name

@syntax.node('name, intr_simplify_fn')
class Intrinsic(syntax.Node):
    def repr(self, ctx):
        return '<intrinsic-fn %s>' % self.name
    def __repr__(self):
        return '<intrinsic-fn %s>' % self.name

@syntax.node('opcode, *args')
class Instruction(syntax.Node):
    def repr(self, ctx):
        return '<instruction %s>(%s)' % (self.opcode, ', '.join(a.repr(ctx)
                for a in self.args))
    def __repr__(self):
        return '<instruction %s>(%s)' % (self.opcode, ', '.join(repr(a)
                for a in self.args))

@syntax.node('&base, scale, ?index, disp')
class Address(syntax.Node):
    def repr(self, ctx):
        return '<address %s+%s*%s+%s>' % (self.base, self.scale, self.index, self.disp)

@syntax.node('value')
class Literal(syntax.Node):
    def repr(self, ctx):
        return '<literal %s>' % self.value

BLOCK_ID = 0
@syntax.node('name, stmts, preds, ?test, succs, #live_ins, #exit_states')
class BasicBlock(syntax.Node):
    def setup(self):
        global BLOCK_ID
        self.block_id = BLOCK_ID
        BLOCK_ID += 1
        self.name = '%s$%s' % (self.name or 'block', self.block_id)
    def __repr__(self):
        return '<block %s>' % self.name

# Just a dumb helper because our @node() decorator doesn't support keyword
# args or defaults
def basic_block(name=None, stmts=None, preds=None, test=None, succs=None,
        live_ins=None, exit_states=None):
    return BasicBlock(name, stmts or [], preds or [], test, succs or [],
            live_ins or {}, exit_states or {}, info=BI)

@syntax.node('#args')
class PhiR(syntax.Node):
    def repr(self, ctx):
        return '<PhiR %s>' % self.args

@syntax.node('name')
class PhiSelect(syntax.Node):
    def repr(self, ctx):
        return '<PhiSelect %s>' % self.name
    def __repr__(self):
        return '<PhiSelect %s>' % self.name

################################################################################
## Graph stuff #################################################################
################################################################################

# Dirty code alert! Throughout the compiler (i.e. the code here, not in the
# bootstrap interpreter), we need basic hashing/equality of nodes, using the
# default Python object behavior (based on reference equality only). But all
# the bootstrap interpreter, for better or worse, was written to override
# __hash__/__eq__ for Mutagen-level equality. Python doesn't allow customization
# of hashing/equality on a per-dict basis, so we originally solved this problem
# with wrapper objects around each node. That sucks, though, so instead we use
# the dynamic nature of Python to do something pretty awful: we just delete
# the overridden functions! The eq/hash functionality here and in the interpreter
# are pretty much mutually exclusive (AFAIK...?), and should stay that way, so
# this is fairly safe. #YOLO

for nt in syntax.ALL_NODE_TYPES:
    if '__eq__' in nt.__dict__:
        del nt.__eq__
        del nt.__hash__

# Machinery for 'smart' nodes in the IR graph, that track uses and can replace
# a given node in all nodes that use it.

class Usage:
    def __init__(self, user, edge_name, type, index=None):
        self.user = user
        self.type = type
        self.edge_name = edge_name
        self.index = index

    def key(self):
        # Use a wrapper for the user here. More annoying stuff to handle
        # __hash__ not being overrideable on a dict-by-dict basis
        return (self.user, self.type, self.edge_name, self.index)

    def __repr__(self):
        return str((self.user, self.type, self.edge_name, self.index))

def add_use(usage, node):
    node._users[usage.key()] = usage

@add_to(syntax.Node)
def forward(self, new_value):
    while self._users:
        [_, usage] = self._users.popitem()
        if usage.type in {ArgType.EDGE, ArgType.OPT}:
            assert getattr(usage.user, usage.edge_name) is self
            setattr(usage.user, usage.edge_name, new_value)
        else:
            container = getattr(usage.user, usage.edge_name)
            assert container[usage.index] is self
            container[usage.index] = new_value

        add_use(usage, new_value)

    remove_children(self)

def remove_children(node):
    for child in node.iterate_children():
        child._users = {key: usage for [key, usage] in child._users.items()
                if usage.user is not node}

def set_edge(node, name, value):
    usage = Usage(node, name, ArgType.EDGE)
    add_use(usage, value)
    setattr(node, name, value)

def remove_edge(node, name):
    usage = Usage(node, name, ArgType.EDGE)
    old_node = getattr(node, name)
    if old_node is not None:
        del old_node._users[usage.key()]
        setattr(node, name, None)

# Unused right now
#def append_to_edge_list(node, name, item):
#    container = getattr(node, name)
#    usage = Usage(node, name, ArgType.LIST, index=len(container))
#    container.append(item)
#    add_use(usage, item)

def set_edge_index(node, edge_name, index, value):
    l = getattr(node, edge_name)
    usage = Usage(node, edge_name, ArgType.LIST, index=index)
    del l[index]._users[usage.key()]
    l[index] = value
    add_use(usage, value)

def set_edge_key(node, edge_name, key, value):
    d = getattr(node, edge_name)
    usage = Usage(node, edge_name, ArgType.DICT, index=key)
    if key in d:
        del d[key]._users[usage.key()]
    d[key] = value
    add_use(usage, value)

def remove_dict_key(node, edge_name, key):
    d = getattr(node, edge_name)
    usage = Usage(node, edge_name, ArgType.DICT, index=key)
    del d[key]._users[usage.key()]
    del d[key]

def clear_edge_dict(node, edge_name):
    d = getattr(node, edge_name)
    for key in d:
        usage = Usage(node, edge_name, ArgType.DICT, index=key)
        del d[key]._users[usage.key()]
    d.clear()

def add_node_usages(node):
    # For every node that this node uses, add a Usage object to its _users list.
    for (arg_type, arg_name) in type(node).arg_defs:
        child = getattr(node, arg_name)
        if arg_type in {ArgType.EDGE, ArgType.OPT}:
            if child is not None:
                add_use(Usage(node, arg_name, arg_type), child)
        elif arg_type == ArgType.LIST:
            for [index, item] in enumerate(child):
                add_use(Usage(node, arg_name, arg_type, index=index), item)
        elif arg_type == ArgType.DICT:
            for [index, item] in child.items():
                add_use(Usage(node, arg_name, arg_type, index=index), item)

# Transform a node and its subtree into a neat graph. This doesn't traverse
# scopes, but returns a list of the scoped expressions it reached.
def transform_to_graph(node):
    functions = []
    # Iterate the graph in topological order
    for node in node.iterate_graph(preorder=False, blacklist=syntax.Scope):
        add_node_usages(node)
        if isinstance(node, syntax.Scope):
            # Only functions for now, no classes/comprehensions
            assert isinstance(node.expr, syntax.Function)
            functions.append(node.expr)
            # Ugh: move the extra args from outside the function's scope into the
            # function itself. We started from a tree so this should always be
            # fine (only one scope per function). These extra args are added to
            # the beginning of the parameter list when compiling the function.
            assert not hasattr(node.expr, 'extra_args')
            # HACK: skip intrinsics for this
            node.extra_args = [arg for arg in node.extra_args
                    if arg not in INTRINSICS]
            node.expr.extra_args = node.extra_args

            # Also, set up attributes for random metadata we might want to store
            node.expr.attributes = {}

    return functions

# Flatten a subgraph below a node into a statements list, and set up node tracking
def create_subgraph(node, statements):
    # Iterate the graph in topological order
    for node in node.iterate_graph(preorder=False, blacklist=syntax.Scope):
        if node._users:
            continue
        statements.append(node)
        add_node_usages(node)

# Node/block copy functions. Each function takes a prefix to modify phi names with,
# so that the names can stay unique when moved into a new namespace

def copy_node(node, node_map, prefix, stmts=None):
    if isinstance(node, syntax.Function):
        node_map[node] = node
        return node
    if node in node_map:
        return node_map[node]

    args = []
    for [arg_type, arg_name] in type(node).arg_defs:
        child = getattr(node, arg_name)
        if arg_type == ArgType.REG:
            assert isinstance(child, (int, str, lir.Inst)), child
            args.append(child)
        elif arg_type in {ArgType.EDGE, ArgType.OPT}:
            args.append(node_map[child])
        elif arg_type == ArgType.LIST:
            args.append([node_map[i] for i in child])
        elif arg_type == ArgType.DICT:
            args.append({k: node_map[v] for [k, v] in child.items()})

    new_node = type(node)(*args, info=node)
    add_node_usages(new_node)
    if isinstance(new_node, PhiSelect):
        new_node.name = prefix + new_node.name
    if stmts is not None:
        stmts.append(new_node)
    node_map[node] = new_node
    return new_node

def copy_block(block, node_map, prefix):
    if block in node_map:
        return node_map[block]
    new_block = basic_block()
    node_map[block] = new_block

    for [name, node] in block.live_ins.items():
        node = copy_node(node, node_map, prefix)
        set_edge_key(new_block, 'live_ins', prefix + name, node)

    new_block.stmts = []
    for stmt in block.stmts:
        copy_node(stmt, node_map, prefix, stmts=new_block.stmts)

    if block.test:
        set_edge(new_block, 'test', node_map[block.test])

    for [name, node] in block.exit_states.items():
        node = copy_node(node, node_map, prefix)
        set_edge_key(new_block, 'exit_states', prefix + name, node)

    new_block.preds = [copy_block(pred, node_map, prefix) for pred in block.preds]
    new_block.succs = [copy_block(succ, node_map, prefix) for succ in block.succs]

    return new_block

################################################################################
## Intrinsics ##################################################################
################################################################################

INTRINSICS = {}

def create_intrinsic(name, fn, arg_types):
    # Create a wrapper that does some common argument checking. Since we can't be
    # sure all arguments are simplified enough to be their final types, we can't
    # throw errors here, but simply fail the simplification.
    def intr_simplify_fn(node):
        if arg_types is not None:
            if len(node.args) != len(arg_types):
                if any(isinstance(arg, syntax.VarArg) for arg in node.args):
                    return None
                node.error('wrong number of arguments to %s, got %s, expected %s' %
                        (name, len(node.args), len(arg_types)))

            for a, t in zip(node.args, arg_types):
                if t is not None and not isinstance(a, t):
                    return None

        return fn(node, *node.args)

    INTRINSICS[name] = Intrinsic(name, intr_simplify_fn, info=BI)
    return INTRINSICS[name]

def mg_intrinsic(arg_types):
    def decorate(fn):
        name = fn.__name__.replace('mgi_', '')
        return create_intrinsic(name, fn, arg_types)
    return decorate

@mg_intrinsic([syntax.Integer, syntax.Integer, syntax.Integer])
def mgi_range(node, a, b, c):
    return syntax.List([syntax.Integer(i, info=node)
            for i in range(a.value, b.value, c.value)], info=node)

@mg_intrinsic([syntax.List])
def mgi_len(node, l):
    return syntax.Integer(len(l.items), info=node)

@mg_intrinsic([syntax.String])
def mgi__extern_label(node, label):
    return ExternSymbol('_' + label.value, info=node)

@mg_intrinsic([syntax.Node])
def mgi_address(node, expr):
    return Address(expr, 0, None, 0, info=node)

@mg_intrinsic([syntax.Function])
def mgi_export(node, fn):
    fn.attributes['export'] = True
    return fn

@mg_intrinsic([syntax.List])
def mgi_static_data(node, l):
    if not all(isinstance(a, syntax.Integer) for a in l):
        return None
    return Literal(create_data(l, 'B'), info=node)

@mg_intrinsic([syntax.String])
def mgi_static_string(node, s):
    data = s.value.encode('utf-8')
    return Literal(lir.literal(asm.Data(data)), info=node)

def create_data(items, dtype):
    data = struct.pack('<' + dtype * len(items), *[i.value for i in items])
    return lir.literal(asm.Data(data))

# Instruction wrappers

def add_inst_instrinsic(spec):
    arg_types = []
    # XXX manual blacklist of instructions that use flags/stack/control flow
    if spec.inst in {*asm.jump_table, *asm.setcc_table, *asm.cmov_table,
            'jmp', 'call', 'ret', 'push', 'pop'}:
        return

    for form in spec.forms:
        if not spec.is_destructive and spec.needs_register:
            form = form[1:]

        # We don't support fixed arguments here. This is stuff like shifts by the
        # cl register
        if any(isinstance(arg, asm.ASMObj) for arg in form):
            continue

        # Only GPR/vector register/immediate operands, no memory
        if all(t in {asm.GPReg, asm.VecReg, asm.Immediate} for [t, s] in form):
            fn = lambda node, *args: Instruction(spec.inst, list(args), info=node)
            # XXX do real arg checking
            create_intrinsic(spec.inst, fn, [None] * len(form))
            return

for spec in asm.INST_SPECS.values():
    add_inst_instrinsic(spec)

# Vector literal instrinsics. XXX range checking

create_intrinsic('vset1_u8', lambda node, arg:
        Literal(lir.Inst('vpbroadcastb', create_data([arg], 'B')), info=node),
        [syntax.Integer])

create_intrinsic('vset32_u8', lambda node, *args:
        Literal(lir.Inst('vmovdqu', create_data(args, 'B')), info=node),
        [syntax.Integer] * 32)

create_intrinsic('vset1_u32', lambda node, arg:
        Literal(lir.Inst('vpbroadcastd', create_data([arg], 'I')), info=node),
        [syntax.Integer])

create_intrinsic('vset8_u32', lambda node, *args:
        Literal(lir.Inst('vmovdqu', create_data(args, 'I')), info=node),
        [syntax.Integer] * 8)

################################################################################
## CFG stuff ###################################################################
################################################################################

def link_blocks(pred, succ):
    pred.succs.append(succ)
    succ.preds.append(pred)

def walk_blocks(block):
    work_list = [block]
    seen = {block}
    while work_list:
        block = work_list.pop(0)
        yield block
        new = [succ for succ in block.succs if succ not in seen]
        work_list.extend(new)
        seen.update(new)

def print_blocks(fn):
    print()
    print(fn.name)
    for block in walk_blocks(fn.first_block):
        print('Block', block.block_id)
        print('  preds:', ' '.join([str(b.block_id) for b in block.preds]))
        print('  succs:', ' '.join([str(b.block_id) for b in block.succs]))
        print('  ins:', ' '.join(sorted(block.live_ins)))
        if block.stmts:
            print('  stmts:')
            for stmt in block.stmts:
                print('    ', hex(id(stmt)), stmt.repr(None))
        if block.test:
            print('  test', block.test.repr(None))
        print('  outs:', ' '.join(sorted(block.exit_states)))

@add_to(syntax.Node)
def gen_blocks(self, current, exit_block):
    for child in self.iterate_children():
        if child and not isinstance(child, syntax.Scope):
            current = child.gen_blocks(current, exit_block)
    return current

@add_to(syntax.Block)
def gen_blocks(self, current, exit_block):
    for stmt in self.stmts:
        current = stmt.gen_blocks(current, exit_block)
        # Hacky way to check for default implementation... Any node that doesn't
        # override gen_blocks() is a "normal" node as far as the CFG is concerned
        if type(stmt).gen_blocks is syntax.Node.gen_blocks:
            current.stmts.append(stmt)
    remove_children(self)
    return current

@add_to(syntax.Return)
def gen_blocks(self, current, exit_block):
    # Self expression is an essential aspect of the human experience
    if self.expr:
        assign = gen_assign('$return_value', self.expr, self)
        current.stmts.append(assign)
        remove_children(self)
    link_blocks(current, exit_block)
    # XXX what to do here? Execution will never continue past a return, so we
    # can't have any sane CFG structure here. We return None for now, and try
    # to handle that properly anywhere downstream (if we don't, we get a weird
    # exception from trying to use None like a block)
    return None

@add_to(syntax.While)
def gen_blocks(self, current, exit_block):
    first = basic_block()
    last = self.block.gen_blocks(first, exit_block)

    test_block = basic_block()
    test_block_last = self.expr.gen_blocks(test_block, exit_block)
    set_edge(test_block_last, 'test', self.expr)

    end_block = basic_block()
    link_blocks(current, test_block)
    link_blocks(test_block_last, first)
    link_blocks(test_block_last, end_block)
    if last:
        link_blocks(last, test_block)
    remove_children(self)
    return end_block

@add_to(syntax.IfElse)
def gen_blocks(self, current, exit_block):
    current = self.expr.gen_blocks(current, exit_block)
    assert not current.test
    set_edge(current, 'test', self.expr)
    if_first = basic_block()
    if_last = self.if_block.gen_blocks(if_first, exit_block)
    else_first = basic_block()
    else_last = self.else_block.gen_blocks(else_first, exit_block)
    link_blocks(current, if_first)
    link_blocks(current, else_first)
    remove_children(self)
    if if_last or else_last:
        end_block = basic_block()
        if if_last:
            link_blocks(if_last, end_block)
        if else_last:
            link_blocks(else_last, end_block)
        return end_block
    return None

# Conditional expressions are, up until this point, a wrapper around IfElse
# that evaluates to a a variable set in each branch. Now that we're using a
# graph representation, we can generate the CFG for the IfElse, and forward the
# variable, and thus forget about this CondExpr.
@add_to(syntax.CondExpr)
def gen_blocks(self, current, exit_block):
    current = self.if_else.gen_blocks(current, exit_block)
    self.forward(self.result)
    return current

################################################################################
## SSA stuff ###################################################################
################################################################################

def add_phi(block, name, info=None):
    if name in block.live_ins:
        value = block.live_ins[name]
    else:
        value = PhiSelect(name, info=info)
        set_edge_key(block, 'live_ins', name, value)
        set_edge_key(block, 'exit_states', name, value)
    return value

# Look up the current value of a name, or create a phi
def load_name(block, name, info=None):
    if name in block.exit_states:
        value = block.exit_states[name]
    # XXX can't reassign intrinsics unless in the same block?!
    elif name in INTRINSICS:
        value = INTRINSICS[name]
    else:
        value = add_phi(block, name, info=info)
    return value

# Recursively destructure an assignment, like [a, [b, c]] = [0, [1, 2]]
def destructure_target(block, statements, lhs, rhs):
    if isinstance(lhs, str):
        set_edge_key(block, 'exit_states', lhs, rhs)
    elif isinstance(lhs, list):
        # Add an assert to check target length
        len_lhs = syntax.Integer(len(lhs), info=rhs)
        len_rhs = syntax.Call(mgi_len, [rhs], info=rhs)
        assertion = syntax.Assert(syntax.BinaryOp('==', len_lhs, len_rhs))
        create_subgraph(assertion, statements)

        for [i, lhs_i] in enumerate(lhs):
            rhs_i = syntax.GetItem(rhs, syntax.Integer(i, info=rhs))
            create_subgraph(rhs_i, statements)
            destructure_target(block, statements, lhs_i, rhs_i)
    else:
        assert False

def gen_ssa_for_stmt(block, statements, stmt):
    for node in stmt.iterate_graph(preorder=False, blacklist=syntax.Scope):
        # Handle loads
        if isinstance(node, syntax.Identifier):
            value = load_name(block, node.name, info=node)
            node.forward(value)
        # Handle stores
        elif isinstance(node, syntax.Assignment):
            for target in node.target.targets:
                destructure_target(block, statements, target, node.rhs)
            remove_edge(node, 'rhs')
        elif isinstance(node, syntax.Target):
            pass
        elif isinstance(node, syntax.Scope):
            # Only functions for now, no classes/comprehensions
            assert isinstance(node.expr, syntax.Function)
            fn = node.expr

            # Create a partial application if there are variables used from
            # an outer scope
            if node.extra_args:
                statements.append(fn)
                fn = syntax.PartialFunction(fn, [load_name(block, arg, info=node)
                        for arg in node.extra_args])
                add_node_usages(fn)

            node.forward(fn)
            statements.append(fn)
        else:
            statements.append(node)

# Propagate phis backwards through the CFG
def propagate_phi(block, name):
    for pred in block.preds:
        if name not in pred.exit_states:
            value = add_phi(pred, name, info=BI)
            propagate_phi(pred, name)

# Trim unneeded values from exit_states
def trim_unused_phis(fn):
    any_trimmed = False
    for block in walk_blocks(fn.first_block):
        live_outs = {name: value for [name, value] in block.exit_states.items()
                if any(name in succ.live_ins for succ in block.succs) or
                # HACK: preserve return value in exit block live outs
                '$return_value' in name}
        for name in set(block.exit_states) - set(live_outs):
            any_trimmed = True
            remove_dict_key(block, 'exit_states', name)
        block.exit_states = live_outs
    return any_trimmed

def gen_ssa(fn):
    assert isinstance(fn, syntax.Function)
    fn.first_block = basic_block(name='enter_block')
    fn.exit_block = basic_block(name='exit_block')
    # Add a return in the exit block of the special variable $return_value.
    # This is the only return that doesn't get handled by gen_blocks().
    # Other returns set this variable and jump to the exit block.
    ret = syntax.Return(syntax.Identifier('$return_value', info=fn))
    add_node_usages(ret)
    fn.exit_block.stmts.append(ret)

    # Convert the AST into a CFG
    last = fn.block.gen_blocks(fn.first_block, fn.exit_block)

    # Generate SSA for all normal nodes
    for block in walk_blocks(fn.first_block):
        statements = []
        for [i, stmt] in enumerate(block.stmts):
            add_stmt = gen_ssa_for_stmt(block, statements, stmt)

        if block.test:
            gen_ssa_for_stmt(block, statements, block.test)

        block.stmts = statements

    for block in walk_blocks(fn.first_block):
        for name in block.live_ins:
            propagate_phi(block, name)

    trim_unused_phis(fn)

################################################################################
## Optimization stuff ##########################################################
################################################################################

# Keep a list of functions that need extra simplifications performed, in case
# any data flows down against the normal bottom-up simplification order. Right
# now, this just happens when a function uses a variable from an outer scope and
# the function only appears in one scope, so the value can be lowered into it
FN_WORK_QUEUE = []

def is_atom(expr):
    return isinstance(expr, (syntax.Integer, syntax.String, ExternSymbol,
            Address,
            # XXX is Literal always an atom?
            Literal,
            # XXX Uhh this probably isn't always chill
            Intrinsic))

def is_simple(expr):
    if isinstance(expr, syntax.List):
        return all(is_simple(item) for item in expr)
    return is_atom(expr)

def can_dce(expr):
    return is_simple(expr) or isinstance(expr, (syntax.BinaryOp,
            syntax.PartialFunction, syntax.Function, syntax.VarArg, syntax.GetItem))

# Decorator for defining a simplifier for a certain node type and given
# arguments. This is basically a clean-ish way of doing pattern matching, and
# we store simplfication functions in a dictionary by node type for quick
# lookup. Each simplification function returns either a node, in which case
# the old node is replaced with the returned one, or None, which is a failure
# sentinel when the node can't be simplified.
SIMPLIFIERS = collections.defaultdict(list)
def simplifier(node_type, *node_args):
    def wrap(fn):
        SIMPLIFIERS[node_type].append((node_args, fn))
        return fn
    return wrap

@simplifier(syntax.UnaryOp, '-', syntax.Integer)
def simplify_int_unop(node):
    return syntax.Integer(-node.rhs.value, info=node)

@simplifier(syntax.BinaryOp, None, syntax.Integer, syntax.Integer)
def simplify_int_binop(node):
    # No short circuiting
    if node.type in {'and', 'or'}:
        return None
    [lhs, rhs] = [node.lhs.value, node.rhs.value]
    op = syntax.BINARY_OP_TABLE[node.type]
    result = getattr(lhs, op)(rhs)
    return syntax.Integer(result, info=node)

@simplifier(syntax.BinaryOp, '+', syntax.List, syntax.List)
def simplify_list_add(node):
    [lhs, rhs] = [node.lhs.items, node.rhs.items]
    return syntax.List(lhs + rhs, info=node)

@simplifier(syntax.BinaryOp, '*', syntax.List, syntax.Integer)
def simplify_list_mul(node):
    [lhs, rhs] = [node.lhs.items, node.rhs.value]
    return syntax.List(lhs * rhs, info=node)

@simplifier(syntax.BinaryOp, '+', Address, syntax.Integer)
@simplifier(syntax.BinaryOp, '-', Address, syntax.Integer)
def simplify_address_disp(node):
    addr = node.lhs
    disp = node.rhs.value
    if node.type == '-':
        disp = -disp
    return Address(addr.base, addr.scale, addr.index, addr.disp + disp, info=node)

@simplifier(syntax.BinaryOp, '+', Address, syntax.BinaryOp)
def simplify_address_index(node):
    addr = node.lhs
    if addr.scale:
        return None
    index = node.rhs
    if index.type == '*' and isinstance(index.lhs, syntax.Integer):
        scale = index.lhs.value
        index = index.rhs
        return Address(addr.base, scale, index, addr.disp, info=node)
    return None

@simplifier(syntax.GetItem, syntax.List, syntax.Integer)
def simplify_getitem(node):
    index = node.item.value
    items = node.obj.items
    assert 0 <= index < len(items)
    return items[index]

@simplifier(syntax.Call, None, None)
def simplify_varargs(node):
    if not any(isinstance(arg, syntax.VarArg) and isinstance(arg.expr, syntax.List)
            for arg in node.args):
        return None
    new_args = []
    for arg in node.args:
        if isinstance(arg, syntax.VarArg) and isinstance(arg.expr, syntax.List):
            new_args.extend(arg.expr.items)
        else:
            new_args.append(arg)
    return syntax.Call(node.fn, new_args)

@simplifier(syntax.Call, Intrinsic, None)
def simplify_intr_call(node):
    return node.fn.intr_simplify_fn(node)

@simplifier(syntax.Call, syntax.PartialFunction, None)
def simplify_partial_call(node):
    return syntax.Call(node.fn.fn, node.fn.args + node.args)

@simplifier(syntax.Assert, (syntax.Boolean, syntax.Integer))
def simplify_assert(node):
    if not node.expr.value:
        # XXX this error message sucks, the expression is always "assert 0", since
        # the assertion got simplified enough to be checked
        node.error('Assertion failed: %s' % node.str(None))
    # XXX weird: just return the expression since we need to signify the assert
    # was simplified out, it should get DCE'd
    return node.expr

# Simplify a partial function if it's the only user of the wrapped function,
# putting the extra arguments (and their child expressions) directly into the
# function body.
@simplifier(syntax.PartialFunction, syntax.Function, None)
def simplify_partial_fn(node):
    fn = node.fn
    assert isinstance(fn, syntax.Function)
    if len(fn._users) > 1:
        return None
    [usage] = fn._users.values()
    assert usage.user is node

    # Make sure there are no phis in the expression graphs of the extra args.
    # Maybe someday we'll do this separately for each arg...
    for arg in node.args:
        for child in arg.iterate_graph():
            if isinstance(child, PhiSelect):
                return None

    # Add a new first block that re-defines values from the partial function call
    new_block = basic_block()
    node_map = {}
    prefix = None
    assert len(fn.extra_args) == len(node.args)
    for [name, arg] in zip(fn.extra_args, node.args):
        arg = copy_node(arg, node_map, prefix, stmts=new_block.stmts)
        set_edge_key(new_block, 'exit_states', name, arg)
    link_blocks(new_block, fn.first_block)
    fn.first_block = new_block

    fn.extra_args = []

    # Make sure we mark this function as needing further simplification
    FN_WORK_QUEUE.append(fn)

    return fn

def simplify_node(node):
    # Look up simplification functions for this node
    node_type = type(node)
    if node_type in SIMPLIFIERS:
        # Get the raw node arguments, don't iterate node lists or dicts
        # or anything like that... this is dumb pattern matching
        args = [getattr(node, arg_name)
                for [_, arg_name] in node_type.arg_defs]
        for [pattern_args, fn] in SIMPLIFIERS[node_type]:
            assert len(pattern_args) == len(args)
            for [p_arg, arg] in zip(pattern_args, args):
                if p_arg is None:
                    pass
                elif isinstance(p_arg, (type, tuple)):
                    if not isinstance(arg, p_arg):
                        break
                elif arg != p_arg:
                    break
            # No break: successful match. Run the simplifier
            else:
                result = fn(node)
                if result is not None:
                    return result

# Delete a control flow edge. The predecessor is generally reachable at this point,
# but we might make the successor unreachable, in which case we need to delete it
# and any outgoing edges it has
def delete_cfg_edge(pred, edge_nb, deleted):
    succ = pred.succs.pop(edge_nb)
    assert succ not in deleted
    remove_edge(pred, 'test')
    succ.preds.remove(pred)

    # No incoming edges on the successor block: delete it and recurse
    if not succ.preds:
        clear_edge_dict(succ, 'exit_states')
        # Delete statements in a bottom-up order, making sure they have no users
        remove_edge(succ, 'test')
        for stmt in reversed(succ.stmts):
            assert not stmt._users, (stmt, stmt._users)
            remove_children(stmt)
        clear_edge_dict(succ, 'live_ins')

        deleted.add(succ)

        for edge in reversed(range(len(succ.succs))):
            delete_cfg_edge(succ, edge, deleted)

# Merge two blocks, assuming the control flow edge between them is the only edge
# for either side
def merge_blocks(pred, succ):
    assert pred.succs == [succ]
    assert succ.preds == [pred]
    # Add statements from succ to pred
    pred.stmts.extend(succ.stmts)

    # Patch up phis
    for [name, phi] in succ.live_ins.items():
        phi.forward(pred.exit_states[name])
    clear_edge_dict(succ, 'live_ins')

    # Delete all the old exit states. There might be more live outs of the pred than
    # live ins of the succ, but we don't care (I think)
    clear_edge_dict(pred, 'exit_states')
    # ...and replace with the new exit states
    for [name, value] in succ.exit_states.items():
        set_edge_key(pred, 'exit_states', name, value)
    # ...and clear the successor exit states for refcounting purposes
    clear_edge_dict(succ, 'exit_states')

    # Pred gets the test from succ
    assert not pred.test
    if succ.test:
        set_edge(pred, 'test', succ.test)
        remove_edge(succ, 'test')

    # Fix up pred/succ links
    pred.succs = succ.succs
    for s in pred.succs:
        s.preds = [pred if p is succ else p for p in s.preds]

# We prefix phi names in inlined functions with a unique prefix so they don't
# collide in the parent scope, based on this ID
INLINE_ID = 0
# We also need to create new temporary variables for phis when values are used across
# inlined calls. By the time functions start getting inlined, the identifiers are
# long gone, and furthermore, we might be carrying across values that never were
# assigned to variables, like the value of a() if b() gets inlined in x = a() + b()
PHI_TEMP_ID = 0

# Inline a call at statement number <i> of the given block. This chops the block
# in half, copies the function's blocks, and pastes them in the middle, fixing up
# phis based on the arguments passed and return value. We also have to handle SSA
# use edges that cross the call statement, creating a new phi that propagates
# across all the inlined blocks
def inline_function(block, i, call):
    global INLINE_ID
    fn = call.fn
    # The function should already have SSA generation/simplification run on it
    assert hasattr(fn, 'first_block')

    prefix = '$inl_%s$' % INLINE_ID
    INLINE_ID += 1

    # Copy the inlined function's blocks
    node_map = {}
    node_map[None] = None # dumb hack
    first_block = copy_block(fn.first_block, node_map, prefix)
    exit_block = node_map[fn.exit_block]

    # Split the current basic block in half
    end_stmts = block.stmts[i+1:]
    end_block = basic_block(stmts=end_stmts)
    del block.stmts[i:]
    if block.test:
        set_edge(end_block, 'test', block.test)
        remove_edge(block, 'test')

    # Patch successors' predecessor lists
    end_block.succs = block.succs
    block.succs = []
    for s in end_block.succs:
        s.preds = [end_block if p is block else p for p in s.preds]

    link_blocks(block, first_block)
    link_blocks(exit_block, end_block)

    # Keep track of nodes defined before the split
    first_half_nodes = set(block.live_ins.values())
    first_half_nodes.update(block.stmts)

    # Move exit states from the first half to the last
    for [name, value] in sorted(block.exit_states.items()):
        if name not in first_half_nodes:
            set_edge_key(end_block, 'exit_states', name, value)
            remove_dict_key(block, 'exit_states', name)

    # Patch arguments
    assert not fn.params.var_params
    assert not fn.params.kw_params
    assert not fn.params.kw_var_params
    params = fn.params.params
    if hasattr(fn, 'extra_args'):
        params = fn.extra_args + params
    assert len(params) == len(call.args)
    for [name, value] in zip(params, call.args):
        name = prefix + name
        set_edge_key(block, 'exit_states', name, value)

    # Patch return value
    [return_stmt] = exit_block.stmts
    assert isinstance(return_stmt, syntax.Return)
    return_id = prefix + '$return_value'
    assert return_id in exit_block.exit_states
    return_phi = add_phi(end_block, return_id, info=return_stmt)
    propagate_phi(end_block, return_id)
    remove_children(return_stmt)
    del exit_block.stmts[0]
    call.forward(return_phi)

    # Helper function to create a new phi and propagate it backwards through
    # the inlined function's CFG
    new_phis = {}
    def make_phi(node):
        global PHI_TEMP_ID
        nonlocal new_phis
        if node not in new_phis:
            temp = '$phi_temp_%s' % PHI_TEMP_ID
            PHI_TEMP_ID += 1
            # First, make sure to add the phi to the exit state of the first
            # half of the split block
            set_edge_key(block, 'exit_states', temp, node)

            new_phis[node] = add_phi(end_block, temp, info=node)
            propagate_phi(end_block, temp)
        return new_phis[node]

    # See which nodes in the first half of the block (before the call) are used
    # after the call: we have to create new phi entries for these values. We've
    # already eliminated identifiers, so we need to create new temporary names
    for stmt in (end_block.stmts + [end_block.test]):
        if stmt is None:
            continue
        # This is a bit gross as far as abstractions go...
        for [arg_type, arg_name] in type(stmt).arg_defs:
            if arg_type in {ArgType.EDGE, ArgType.OPT}:
                child = getattr(stmt, arg_name)
                if child is not None and child in first_half_nodes:
                    set_edge(stmt, arg_name, make_phi(child))
            elif arg_type == ArgType.LIST:
                for [index, child] in enumerate(getattr(stmt, arg_name)):
                    if child in first_half_nodes:
                        set_edge_index(stmt, arg_name, index, make_phi(child))
            elif arg_type == ArgType.DICT:
                for [key, child] in getattr(stmt, arg_name).items():
                    if child in first_half_nodes:
                        set_edge_key(stmt, arg_name, key, make_phi(child))

def simplify_fn(fn):
    # Basic simplification pass. We repeatedly run the full simplification until
    # nothing gets simplified, which is a bit wasteful, since simplifications
    # only rarely trigger downstream simplifications, but oh well
    any_simplified = True
    while any_simplified:
        any_simplified = False

        # First pass: run DCE and graph rewriting
        for block in walk_blocks(fn.first_block):
            # Loop over all statements in the block. The statement list can get
            # dynamically modified so use a manual index.
            i = 0
            while i < len(block.stmts):
                node = block.stmts[i]
                # Try to dead-code-eliminate this node
                if not node._users and can_dce(node):
                    any_simplified = True
                    remove_children(node)
                    del block.stmts[i]
                    continue

                # Run local graph simplifications, as defined in the
                # @simplifier()s above
                result = simplify_node(node)
                if result is not None:
                    # Update graph tracking for any sub-nodes of the result
                    new_stmts = []
                    create_subgraph(result, new_stmts)
                    node.forward(result)
                    # Replace the old node with the flattened subgraph of new
                    # nodes in the statements list. We don't update i here, which
                    # means we rerun the simplify pass over the new nodes.
                    block.stmts[i:i+1] = new_stmts
                    any_simplified = True
                    continue

                # Try inlining function calls
                if isinstance(node, syntax.Call) and isinstance(node.fn, syntax.Function):
                    inline_function(block, i, node)

                i += 1
        # Second pass: try to simplify the CFG
        deleted = set()
        for block in walk_blocks(fn.first_block):
            # This should never trigger, since the walk_blocks() generator only 
            # adds a block's successors to the queue after the block has been
            # yielded. Whenever we delete a block, it can only be reached through
            # the current block, either as the only successor (when it's merged)
            # or as an untaken branch (and downstream blocks from there).
            assert block not in deleted

            # Resolve any constant branches
            if isinstance(block.test, (syntax.Boolean, syntax.Integer)):
                assert len(block.succs) == 2
                # For some reason I used 0 is true, 1 is false for conditional branches
                # We want the not-taken branch index, so reverse it
                not_taken = int(block.test.value != 0)
                delete_cfg_edge(block, not_taken, deleted)
                any_simplified = True

            # Merge any blocks that don't have any other predecessors/successors
            if len(block.succs) == 1 and len(block.succs[0].preds) == 1:
                succ = block.succs[0]
                if succ is not fn.exit_block:
                    deleted.add(succ)
                    merge_blocks(block, succ)
                    any_simplified = True

        if trim_unused_phis(fn):
            any_simplified = True

################################################################################
## LIR conversion stuff ########################################################
################################################################################

BINOP_TABLE = {
    '+': lir.add,
    '-': lir.sub,
    '*': lir.mul,
    '&': lir.band,
    '|': lir.bor,
}
CMP_TABLE = {
    '<': 'l',
    '<=': 'le',
    '==': 'e',
    '>=': 'ge',
    '>': 'g',
}

def gen_lir_for_node(block, node, block_map, node_map):
    if isinstance(node, syntax.BinaryOp):
        if node.type in BINOP_TABLE:
            fn = BINOP_TABLE[node.type]
            return fn(node_map[node.lhs], node_map[node.rhs])
        elif node.type in CMP_TABLE:
            cc = CMP_TABLE[node.type]
            return [lir.cmp(node_map[node.lhs], node_map[node.rhs]),
                    lir.Inst('set' + cc)]
    elif isinstance(node, ExternSymbol):
        return lir.literal(asm.ExternLabel(node.name))
    elif isinstance(node, syntax.Integer):
        return lir.literal(node.value)
    elif isinstance(node, syntax.Call):
        return lir.call(node_map[node.fn], *[node_map[arg] for arg in node.args])
    elif isinstance(node, syntax.Function) and node.attributes.get('export'):
        # XXX need to make sure this name is unique
        return lir.literal(asm.ExternLabel('_' + node.name))
    elif isinstance(node, syntax.Return):
        return lir.Return(node_map[node.expr])
    elif isinstance(node, Instruction):
        return lir.Inst(node.opcode, *[node_map[arg] for arg in node.args])
    elif isinstance(node, Address):
        index = node_map[node.index] if node.index else None
        return lir.Address(node_map[node.base], node.scale, index, node.disp)
    # Literal is just a wrapper for raw LIR objects
    elif isinstance(node, Literal):
        return list(node.value.flatten())

    uses = []
    for usage in node._users.values():
        base = '%r.%s' % (usage.user, usage.edge_name)
        if usage.type in {ArgType.LIST, ArgType.DICT}:
            base += '[%r]' % usage.index
        uses.append(base)
    node.error('unsupported node type %s for compilation: %s\nUses:\n  %s' % (
            type(node).__name__, node.repr(None), '\n  '.join(uses)))

def generate_lir(fn):
    node_map = {}
    block_map = {}
    new_blocks = []

    # Create a parameter name->number dictionary for the first block
    assert not fn.params.var_params
    assert not fn.params.kw_params
    assert not fn.params.kw_var_params
    params = fn.params.params
    if hasattr(fn, 'extra_args'):
        params = fn.extra_args + params
    params = {name: i for [i, name] in enumerate(params)}

    # Generate LIR blocks, and the write portion of the phis (this is for
    # variables that are used before they are defined in a block, aka live ins).
    for block in walk_blocks(fn.first_block):
        phi_write = lir.PhiW(None, {} if block is fn.first_block else block.live_ins)
        phi_selects = {}
        insts = []
        for [name, live_in] in sorted(block.live_ins.items()):
            # For the first block, create parameter LIR objects to bind to the
            # parameter names directly in place of PhiSelects
            if block is fn.first_block:
                assert name in params, (name, params)
                value = lir.parameter(params[name])
            else:
                assert isinstance(live_in, PhiSelect)
                value = lir.PhiSelect(phi_write, name)
                phi_selects[name] = value

            node_map[live_in] = value
            insts.append(value)

        # Create the phi read with MIR nodes, we'll fix it up after all
        # LIR nodes are created
        phi_read = lir.PhiR(block.exit_states)

        b = lir.BasicBlock(block.name, phi_write, phi_selects,
                insts, None, phi_read, block.preds, block.succs)
        new_blocks.append(b)
        block_map[block] = b

    # Generate LIR for all normal nodes
    for block in walk_blocks(fn.first_block):
        lir_block = block_map[block]
        for stmt in block.stmts:
            # HACK to get around duplicated nodes in the statement list
            # XXX should also check for block boundaries etc...
            if stmt in node_map:
                continue
            insts = gen_lir_for_node(block, stmt, block_map, node_map)
            if not isinstance(insts, list):
                insts = [insts]
            lir_block.insts.extend(insts)
            node_map[stmt] = insts[-1]

        test = None
        if block.test:
            n = node_map[block.test]
            lir_block.test = lir.test(n, n)
            lir_block.insts.append(lir_block.test)

    # Fix up CFG: look up LIR blocks for preds/succs, now that they've all
    # been instantiated
    for block in new_blocks:
        block.preds = [block_map[b] for b in block.preds]
        block.succs = [block_map[b] for b in block.succs]

    # Same deal with phis: now that we've constructed all the LIR objects, look
    # up the values of the phi arguments and replace them.
    for block in new_blocks:
        block.phi_write.phi_reads = [pred.phi_read
                for pred in block.preds]
        block.phi_read.args = {k: node_map[v]
                for [k, v] in block.phi_read.args.items()}

    return new_blocks

################################################################################
## Compilation mgmt stuff ######################################################
################################################################################

# Recursively compile a function, that is, transform the parsed AST representation
# of a function and any nested functions within it to SSA and pass it through
# our basic optimization passes. This doesn't do the final register allocator or
# assembler stages, which are performed later after all functions are collected
# and optimized, since the usage of a function in parent scopes can change how
# it's compiled (like the @export decorator, without which the function won't
# actually be converted to assembly)
def compile_fn(fn):
    # Convert to graph representation, collecting nested functions
    extra_functions = transform_to_graph(fn)

    # Recursively compile any nested functions first
    for extra_fn in extra_functions:
        yield from compile_fn(extra_fn)

    gen_ssa(fn)

    simplify_fn(fn)

    yield fn

def compile_stmts(stmts):
    global FN_WORK_QUEUE

    # Wrap top-level statements in a main function
    block = syntax.Block(stmts, info=BI)
    main_params = syntax.Params(['argc', 'argv'], [], None, [], None, info=BI)
    main_fn = syntax.Function('main', main_params, None, block)
    main_fn.attributes = {'export': True}

    ctx = syntax.Context('__main__', None, None)
    main_fn = parse.preprocess_program(ctx, main_fn, include_io_handlers=False)

    all_fns = list(compile_fn(main_fn))

    # Repeatedly simplify any functions that got modified by their parents
    while FN_WORK_QUEUE:
        work_queue = FN_WORK_QUEUE
        FN_WORK_QUEUE = []
        for fn in work_queue:
            simplify_fn(fn)

    return all_fns

def compile_file(input_path):
    try:
        stmts = parse.parse_file(input_path, import_builtins=False)

        return compile_stmts(stmts)
    except (libparse.ParseError, syntax.ProgramError) as e:
        e.print()
        sys.exit(1)

def print_insts(fn):
    print()
    for inst in fn.insts:
        if isinstance(inst, asm.Label):
            print('{}:'.format(inst.name))
        else:
            print('    {}'.format(inst))

# Transform the final mostly-optimized MIR into LIR, which maps pretty much
# 1-to-1 with instructions, and allocate registers, returning a list of assembly
# instructions
def export_fn(fn):
    lir_blocks = generate_lir(fn)

    lir_fn = lir.Function('_' + fn.name, fn.params.params, lir_blocks,
            attributes=fn.attributes)

    return regalloc.allocate_registers(lir_fn)

# Build the assembly representation of a list of functions and write the bytes
# out to an ELF file
def export_functions_elf(path, fns):
    all_insts = []
    for fn in fns:
        fn.insts = export_fn(fn)
        all_insts.extend(fn.insts)

    elf_file = elf.create_elf_file(*asm.build(all_insts))
    with open(path, 'wb') as f:
        f.write(bytes(elf_file))

def main(args):
    fns = compile_file(args[1])
    exported_fns = [fn for fn in fns if fn.attributes.get('export')]
    export_functions_elf('elfout.o', fns)

if __name__ == '__main__':
    main(sys.argv)
