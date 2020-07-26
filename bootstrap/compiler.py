#!/usr/bin/env python3
import collections.abc
import sys

import sprdpl.parse as libparse

import asm
import lir
import parse
import regalloc
import syntax
from syntax import ArgType

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

@syntax.node('index')
class Parameter(syntax.Node):
    def repr(self, ctx):
        return 'Param(%s)' % self.index

@syntax.node('name')
class ExternSymbol(syntax.Node):
    def repr(self, ctx):
        return 'ExternSymbol(%s)' % self.name

@syntax.node('name, simplify_fn')
class Intrinsic(syntax.Node):
    def repr(self, ctx):
        return '<intrinsic-fn %s>' % self.name

@syntax.node('opcode, *args')
class Instruction(syntax.Node):
    def repr(self, ctx):
        return '<instruction %s>(%s)' % (self.opcode, ', '.join(a.repr(ctx)
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
@syntax.node('stmts, *preds, ?test, *succs, #live_ins, #exit_states')
class BasicBlock(syntax.Node):
    def setup(self):
        global BLOCK_ID
        self.block_id = BLOCK_ID
        BLOCK_ID += 1

# Just a dumb helper because our @node() decorator doesn't support keyword
# args or defaults
def basic_block(stmts=None, preds=None, test=None, succs=None,
        live_ins=None, exit_states=None):
    return BasicBlock(stmts or [], preds or [], test, succs or [],
            live_ins or {}, exit_states or {}, info=BI)

@syntax.node('#args')
class PhiR(syntax.Node):
    def repr(self, ctx):
        return '<PhiR %s>' % self.args

@syntax.node('name')
class PhiSelect(syntax.Node):
    def repr(self, ctx):
        return '<PhiSelect %s>' % self.name

################################################################################
## Graph stuff #################################################################
################################################################################

# Here we have a dumb custom dict class to be able to hash nodes based on just
# their identity and not use the regular __hash__/__eq__ machinery (which works
# with Mutagen semantics, possibly calling user code, since we need a fast dict
# implementation in the bootstrap).

# Basic wrapper for every node to override its __hash__/__eq__. Sucks
# that we have this O(n) memory overhead...
class NodeWrapper:
    def __init__(self, node):
        assert isinstance(node, syntax.Node), '%s: %s' % (type(node), repr(node))
        self.node = node
    def __hash__(self):
        return id(self.node)
    def __eq__(self, other):
        return self.node is other.node
    def __repr__(self):
        return 'NW(%s)' % repr(self.node)

# Dictionary that wraps every key with a NodeWrapper
class NodeDict(collections.abc.MutableMapping):
    def __init__(self, *args, **kwargs):
        self._items = {}
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self._items[NodeWrapper(key)]
    def __setitem__(self, key, value):
        self._items[NodeWrapper(key)] = value
    def __delitem__(self, key):
        del self._items[NodeWrapper(key)]

    def __iter__(self):
        for key in self._items:
            yield key.node
    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return repr(self._items)

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
        return (NodeWrapper(self.user), self.type, self.edge_name, self.index)


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
    # Iterate the graph in reverse depth-first order to get a topological ordering
    for node in reversed(list(node.iterate_graph(blacklist=syntax.Scope))):
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
            node.expr.extra_args = node.extra_args

    # Un-reverse the functions before returning
    return functions[::-1]

################################################################################
## Intrinsics ##################################################################
################################################################################

INTRINSICS = {}

def create_intrinsic(name, fn, arg_types):
    # Create a wrapper that does some common argument checking. Since we can't be
    # sure all arguments are simplified enough to be their final types, we can't
    # throw errors here, but simply fail the simplification.
    def simplify_fn(node):
        if arg_types is not None:
            if len(node.args) != len(arg_types):
                return None
            for a, t in zip(node.args, arg_types):
                if t is not None and not isinstance(a, t):
                    return None

        return fn(node, *node.args)

    INTRINSICS[name] = Intrinsic(name, simplify_fn, info=BI)

def mg_intrinsic(arg_types):
    def decorate(fn):
        name = fn.__name__.replace('mgi_', '')
        create_intrinsic(name, fn, arg_types)
        return None
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

@mg_intrinsic([syntax.List])
def mgi_static_data(node, l):
    if not all(isinstance(a, syntax.Integer) for a in l):
        return None
    return Literal(create_data(l), info=node)

def create_data(items):
    return lir.literal(asm.Data([i.value for i in items]))

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
    Literal(lir.Inst('vpbroadcastb', create_data([arg])), info=node), [syntax.Integer])

create_intrinsic('vset32_u8', lambda node, *args:
    Literal(lir.Inst('vmovdqu', create_data(args)), info=node), [syntax.Integer] * 32)

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

def print_blocks(fn, first_block):
    print()
    print(fn.name)
    for block in walk_blocks(first_block):
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
    return current

@add_to(syntax.Return)
def gen_blocks(self, current, exit_block):
    # Self expression is an essential aspect of the human experience
    if self.expr:
        assign = gen_assign('$return_value', self.expr, self)
        current.stmts.append(assign)
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

    exit_block = basic_block()
    link_blocks(current, test_block)
    link_blocks(test_block_last, first)
    link_blocks(test_block_last, exit_block)
    if last:
        link_blocks(last, test_block)
    return exit_block

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
    if if_last or else_last:
        exit_block = basic_block()
        if if_last:
            link_blocks(if_last, exit_block)
        if else_last:
            link_blocks(else_last, exit_block)
        return exit_block
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

def destructure_target(block, statements, lhs, rhs):
    if isinstance(lhs, str):
        set_edge_key(block, 'exit_states', lhs, rhs)
    elif isinstance(lhs, list):
        # XXX need to check target length
        for [i, lhs_i] in enumerate(lhs):
            rhs_i = syntax.GetItem(rhs, syntax.Integer(i, info=rhs))
            add_node_usages(rhs_i)
            statements.append(rhs_i)
            destructure_target(block, statements, lhs_i, rhs_i)
    else:
        assert False

def gen_ssa_for_stmt(block, statements, stmt):
    for node in reversed(list(stmt.iterate_graph(blacklist=syntax.Scope))):
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
            # XXX need to make sure this name is unique
            label = ExternSymbol('_' + node.expr.name, info=node)

            # Create a partial application if there are variables used from
            # an outer scope
            if node.extra_args:
                statements.append(label)
                fn = syntax.PartialFunction(label, [load_name(block, arg, info=node)
                        for arg in node.extra_args])
                add_node_usages(fn)
            else:
                fn = label

            node.forward(fn)
            statements.append(fn)
        else:
            statements.append(node)

def gen_ssa(fn):
    # First off, generate a CFG
    assert isinstance(fn, syntax.Function)
    first_block = basic_block()
    exit_block = basic_block()
    # Add a return in the exit block of the special variable $return_value.
    # This is the only return that doesn't get handled by gen_blocks().
    # Other returns set this variable and jump to the exit block.
    ret = syntax.Return(syntax.Identifier('$return_value', info=fn))
    add_node_usages(ret)
    exit_block.stmts.append(ret)
    last = fn.block.gen_blocks(first_block, exit_block)

    # Generate SSA for all normal nodes
    for block in walk_blocks(first_block):
        statements = []
        for [i, stmt] in enumerate(block.stmts):
            add_stmt = gen_ssa_for_stmt(block, statements, stmt)

        if block.test:
            gen_ssa_for_stmt(block, statements, block.test)

        block.stmts = statements

    # Propagate phis backwards through the CFG
    def propagate_phi(block, name):
        for pred in block.preds:
            if name not in pred.exit_states:
                value = add_phi(pred, name, info=BI)
                propagate_phi(pred, name)

    for block in walk_blocks(first_block):
        for name in block.live_ins:
            propagate_phi(block, name)

    # Trim unneeded values from exit_states
    for block in walk_blocks(first_block):
        live_outs = {name: value for [name, value] in block.exit_states.items()
            if any(name in succ.live_ins for succ in block.succs)}
        for name in set(block.exit_states) - set(live_outs):
            remove_dict_key(block, 'exit_states', name)
        block.exit_states = live_outs

    return first_block

################################################################################
## Optimization stuff ##########################################################
################################################################################

def is_atom(expr):
    return isinstance(expr, (syntax.Integer, syntax.String, ExternSymbol,
            Address,
            # XXX is Literal always an atom?
            Literal))

def is_atom_list(expr):
    return isinstance(expr, syntax.List) and all(is_atom(item) for item in expr)

def can_dce(expr):
    return is_atom(expr) or is_atom_list(expr) or isinstance(expr, (
            syntax.BinaryOp, syntax.PartialFunction, syntax.VarArg))

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
    if not any(isinstance(arg, syntax.VarArg) and is_atom_list(arg.expr)
            for arg in node.args):
        return None
    new_args = []
    for arg in node.args:
        if isinstance(arg, syntax.VarArg) and is_atom_list(arg.expr):
            new_args.extend(arg.expr.items)
        else:
            new_args.append(arg)
    return syntax.Call(node.fn, new_args)

@simplifier(syntax.Call, Intrinsic, None)
def simplify_intr_call(node):
    return node.fn.simplify_fn(node)

@simplifier(syntax.Call, syntax.PartialFunction, None)
def simplify_partial_call(node):
    return syntax.Call(node.fn.fn, node.fn.args + node.args)

def simplify_blocks(first_block):
    # Basic simplification pass. We repeatedly run the full simplification until
    # nothing gets simplified, which is a bit wasteful, since simplifications
    # only rarely trigger downstream simplifications, but oh well
    any_simplified = True
    while any_simplified:
        any_simplified = False

        for block in walk_blocks(first_block):
            for [i, stmt] in enumerate(block.stmts):
                # Look up simplification functions for this node
                node_type = type(stmt)
                if node_type in SIMPLIFIERS:
                    # Get the raw node arguments, don't iterate node lists or dicts
                    # or anything like that... this is dumb pattern matching
                    args = [getattr(stmt, arg_name)
                            for [_, arg_name] in node_type.arg_defs]
                    for [pattern_args, fn] in SIMPLIFIERS[node_type]:
                        assert len(pattern_args) == len(args)
                        for [p_arg, arg] in zip(pattern_args, args):
                            if p_arg is None:
                                pass
                            elif isinstance(p_arg, type):
                                if not isinstance(arg, p_arg):
                                    break
                            elif arg != p_arg:
                                break
                        # No break: successful match. Run the simplifier
                        else:
                            result = fn(stmt)
                            if result is not None:
                                # Update graph tracking for any sub-nodes of
                                # the result.
                                # XXX This only works if the only new node created
                                # is the result. If new nested nodes were to be
                                # created, they'd have to get inserted into the
                                # block.stmts list, which is messy...
                                add_node_usages(result)
                                stmt.forward(result)
                                block.stmts[i] = result
                                remove_children(stmt)
                                any_simplified = True

    # Run DCE
    any_removed = True
    while any_removed:
        any_removed = False
        for block in walk_blocks(first_block):
            statements = []
            for [i, stmt] in enumerate(block.stmts):
                if can_dce(stmt) and not stmt._users:
                    any_removed = True
                    remove_children(stmt)
                else:
                    statements.append(stmt)

            block.stmts = statements

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
    elif isinstance(node, Parameter):
        return lir.parameter(node.index)
    elif isinstance(node, ExternSymbol):
        return lir.literal(asm.ExternLabel(node.name))
    elif isinstance(node, syntax.Integer):
        return lir.literal(node.value)
    elif isinstance(node, syntax.Call):
        return lir.call(node_map[node.fn], *[node_map[arg] for arg in node.args])
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
    assert False, str(node)

def block_name(block):
    return 'block$%s' % block.block_id

def generate_lir(first_block):
    node_map = NodeDict()
    block_map = NodeDict()
    new_blocks = []

    # Generate LIR blocks, and the write portion of the phis (this is for
    # variables that are used before they are defined in a block, aka live ins).
    for block in walk_blocks(first_block):
        phi_write = lir.PhiW(None, block.live_ins)
        phi_selects = {}
        insts = []
        for [name, live_in] in sorted(block.live_ins.items()):
            sel = lir.PhiSelect(phi_write, name)
            phi_selects[name] = sel
            node_map[live_in] = sel
            insts.append(sel)

        # Create the phi read with MIR nodes, we'll fix it up after all
        # LIR nodes are created
        phi_read = lir.PhiR(block.exit_states)

        b = lir.BasicBlock(block_name(block), phi_write, phi_selects,
                insts, None, phi_read, block.preds, block.succs)
        new_blocks.append(b)
        block_map[block] = b

    # Generate LIR for all normal nodes
    for block in walk_blocks(first_block):
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

def compile_fn(fn):
    # Create argument bindings
    assert not fn.params.var_params
    assert not fn.params.kw_params
    assert not fn.params.kw_var_params
    prologue = []
    params = fn.params.params
    if hasattr(fn, 'extra_args'):
        params = fn.extra_args + params
    for [i, name] in enumerate(params):
        prologue.append(gen_assign(name, Parameter(i, info=fn), fn))
    fn.block.stmts = prologue + fn.block.stmts

    # Convert to graph representation, collecting nested functions
    extra_functions = transform_to_graph(fn)

    # Recursively compile any nested functions first
    for extra_fn in extra_functions:
        yield from compile_fn(extra_fn)

    first_block = gen_ssa(fn)

    simplify_blocks(first_block)

    lir_blocks = generate_lir(first_block)

    # Add the final LIR function to the main list of functions
    yield lir.Function('_' + fn.name, fn.params.params, lir_blocks)

def compile_stmts(stmts):
    # Wrap top-level statements in a main function
    block = syntax.Block(stmts, info=BI)
    main_params = syntax.Params(['argc', 'argv'], [], None, [], None, info=BI)
    main_fn = syntax.Function('main', main_params, None, block)

    ctx = syntax.Context('__main__', None, None)
    main_fn = parse.preprocess_program(ctx, main_fn, include_io_handlers=False)

    return list(compile_fn(main_fn))

def compile_file(path):
    try:
        stmts = parse.parse_file(path, import_builtins=False)

        fns = compile_stmts(stmts)
    except (libparse.ParseError, syntax.ProgramError) as e:
        e.print()
        sys.exit(1)

    regalloc.export_functions('elfout.o', fns)

def main(args):
    compile_file(args[1])

if __name__ == '__main__':
    main(sys.argv)
