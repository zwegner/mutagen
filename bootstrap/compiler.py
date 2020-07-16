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
        return '<instruction %s>(%s)' % (self.opcode, self.args)

BLOCK_ID = 0
@syntax.node('*stmts, *preds, ?test, *succs, #live_ins, #exit_states')
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

    def target(self):
        if self.type in {ArgType.EDGE, ArgType.OPT}:
            return getattr(self.user, self.edge_name)
        elif self.type == ArgType.LIST:
            container = getattr(self.user, self.edge_name)
            return container[self.index] if self.index < len(container) else None
        elif self.type == ArgType.DICT:
            container = getattr(self.user, self.edge_name)
            return container.get(self.index)

def add_use(usage, node):
    if usage.index == -1 and usage.type == ArgType.LIST:
        container = getattr(usage.user, usage.edge_name)
        usage.index = len(container)
        container.append(None)

    if usage.type in {ArgType.EDGE, ArgType.OPT}:
        setattr(usage.user, usage.edge_name, node)
    elif usage.type in {ArgType.LIST, ArgType.DICT}:
        container = getattr(usage.user, usage.edge_name)
        container[usage.index] = node

    if node is not None:
        node._uses[usage.key()] = usage

def remove_use(usage):
    old_node = usage.target()
    if old_node is not None:
        del old_node._uses[usage.key()]

def replace_use(usage, node):
    remove_use(usage)
    add_use(usage, node)

@add_to(syntax.Node)
def forward(self, new_value):
    while self._uses:
        [_, usage] = self._uses.popitem()
        assert usage.target() is self
        add_use(usage, new_value)

@add_to(syntax.Node)
def remove_uses_by(self, user):
    self._uses = {key: usage for [key, usage] in self._uses.items()
            if usage.user is not user}

def set_edge(node, name, value):
    # XXX ArgType.EDGE is not necessarily accurate, for now rely on identical handling
    # of EDGE/OPT for Usage, cuz I'm lazy
    replace_use(Usage(node, name, ArgType.EDGE), value)

def append_to_edge_list(node, name, item):
    # Don't need to check for an old node here
    add_use(Usage(node, name, ArgType.LIST, index=-1), item)

def set_edge_key(node, name, key, value):
    replace_use(Usage(node, name, ArgType.DICT, index=key), value)

def add_node_usages(node):
    # For every node that this node uses, add a Usage object to its _uses list.
    for (arg_type, arg_name) in type(node).arg_defs:
        child = getattr(node, arg_name)
        if arg_type in {ArgType.EDGE, ArgType.OPT}:
            if child is not None:
                add_use(Usage(node, arg_name, arg_type), child)
        elif arg_type in {ArgType.LIST, ArgType.DICT}:
            children = enumerate(child) if arg_type == ArgType.LIST else child.items()
            for [index, item] in children:
                add_use(Usage(node, arg_name, arg_type, index=index), item)

def transform_to_graph(block):
    # Iterate the graph in reverse depth-first order to get a topological ordering
    for node in reversed(list(block.iterate_graph())):
        add_node_usages(node)

################################################################################
## Intrinsics ##################################################################
################################################################################

INTRINSICS = {}

def create_intrinsic(name, fn, arg_types):
    # Create a wrapper that does some common argument checking. Since we can't be
    # sure all arguments are simplified enough to be their final types, we can't
    # throw errors here, but simply fail the simplification.
    # For a call node, this function is called like this:
    # call.fn.simplify_fn(call, call.args)
    def simplify(node, args):
        if arg_types is not None:
            if len(args) != len(arg_types):
                node.error('wrong number of arguments')
            for a, t in zip(args, arg_types):
                if t is not None and not isinstance(a, t):
                    return False

        new_node = fn(node, *args)
        if new_node is not None:
            # If the simplify function returns a new node, update the graph tracking.
            # This is not very clean, and there should be a better long term way of
            # keeping the graph up-to-date when creating/deleting nodes, but for now
            # having this isolated just here is acceptable.
            add_node_usages(new_node)
            node.forward(new_node)
            for arg in args:
                arg.remove_uses_by(node)
            return True
        return False

    INTRINSICS[name] = Intrinsic(name, simplify, info=BI)

def mg_intrinsic(arg_types):
    def decorate(fn):
        name = fn.__name__.replace('mgi_', '')
        create_intrinsic(name, fn, arg_types)
        return None
    return decorate

@mg_intrinsic([syntax.String])
def mgi__extern_label(node, label):
    return ExternSymbol('_' + label.value, info=node)

# Instruction wrappers
inst_specs = {
    'lzcnt': 1,
    'popcnt': 1,
    'tzcnt': 1,
    'pext': 2,
    'pdep': 2,
}

def add_inst_instrinsic(opcode, n_args):
    if n_args == 1:
        fn = lambda node, a: Instruction(opcode, [a], info=node)
    elif n_args == 2:
        fn = lambda node, a, b: Instruction(opcode, [a, b], info=node)
    else:
        assert False
    create_intrinsic('_builtin_' + opcode, fn, [None] * n_args)

for inst, n_args in inst_specs.items():
    add_inst_instrinsic(inst, n_args)

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

def print_blocks(block):
    for block in walk_blocks(block):
        print('Block', block.block_id)
        print('  preds:', ' '.join([str(b.block_id) for b in block.preds]))
        print('  succs:', ' '.join([str(b.block_id) for b in block.succs]))
        print('  ins:', ' '.join(sorted(block.live_ins)))
        if block.stmts:
            print('  stmts:')
            for stmt in block.stmts:
                print('    ', stmt.repr(None))
        if block.test:
            print('  test', block.test.repr(None))
        print('  outs:', ' '.join(sorted(block.exit_states)))

@add_to(syntax.Node)
def gen_blocks(self, current):
    for child in self.iterate_children():
        if child:
            current = child.gen_blocks(current)
    return current

@add_to(syntax.Block)
def gen_blocks(self, current):
    for stmt in self.stmts:
        current = stmt.gen_blocks(current)
        # Hacky way to check for default implementation
        if type(stmt).gen_blocks is syntax.Node.gen_blocks:
            append_to_edge_list(current, 'stmts', stmt)
    return current

@add_to(syntax.While)
def gen_blocks(self, current):
    first = basic_block()
    last = self.block.gen_blocks(first)

    test_block = basic_block()
    test_block_last = self.expr.gen_blocks(test_block)
    set_edge(test_block_last, 'test', self.expr)

    exit_block = basic_block()
    link_blocks(current, test_block)
    link_blocks(test_block_last, first)
    link_blocks(test_block_last, exit_block)
    link_blocks(last, test_block)
    return exit_block

@add_to(syntax.IfElse)
def gen_blocks(self, current):
    current = self.expr.gen_blocks(current)
    assert not current.test
    set_edge(current, 'test', self.expr)
    if_first = basic_block()
    if_last = self.if_block.gen_blocks(if_first)
    else_first = basic_block()
    else_last = self.else_block.gen_blocks(else_first)
    exit_block = basic_block()
    link_blocks(current, if_first)
    link_blocks(current, else_first)
    link_blocks(if_last, exit_block)
    link_blocks(else_last, exit_block)
    return exit_block

# Conditional expressions are, up until this point, a wrapper around IfElse
# that evaluates to a a variable set in each branch. Now that we're using a
# graph representation, we can generate the CFG for the IfElse, and forward the
# variable, and thus forget about this CondExpr.
@add_to(syntax.CondExpr)
def gen_blocks(self, current):
    current = self.if_else.gen_blocks(current)
    self.forward(self.result)
    return current

################################################################################
## SSA stuff ###################################################################
################################################################################

def add_phi(block, statements, name, info=None):
    if name in block.live_ins:
        value = block.live_ins[name]
    else:
        value = PhiSelect(name, info=info)
        set_edge_key(block, 'live_ins', name, value)
        set_edge_key(block, 'exit_states', name, value)
    return value

def gen_ssa_for_stmt(block, statements, stmt):
    for node in reversed(list(stmt.iterate_graph())):
        # Handle loads
        if isinstance(node, syntax.Identifier):
            if node.name in block.exit_states:
                value = block.exit_states[node.name]
            elif node.name in INTRINSICS:
                value = INTRINSICS[node.name]
            else:
                value = add_phi(block, statements, node.name, info=node)
            node.forward(value)
        # Handle stores
        elif isinstance(node, syntax.Assignment):
            for target in node.target.targets:
                # XXX destructuring assignment is more complicated, we would
                # need to desugar it to involve temporaries and indexing
                assert isinstance(target, str)
                set_edge_key(block, 'exit_states', target, node.rhs)
            remove_use(Usage(node, 'rhs', ArgType.EDGE))
        elif isinstance(node, syntax.Target):
            pass
        else:
            statements.append(node)

def gen_ssa(block):
    first_block = basic_block()
    last = block.gen_blocks(first_block)

    for block in walk_blocks(first_block):
        statements = []
        for [i, stmt] in enumerate(block.stmts):
            add_stmt = gen_ssa_for_stmt(block, statements, stmt)

            # XXX need to prevent deletion here eventually
            remove_use(Usage(block, 'stmts', ArgType.LIST, index=i))

        if block.test:
            gen_ssa_for_stmt(block, statements, block.test)

        block.stmts = []
        for stmt in statements:
            append_to_edge_list(block, 'stmts', stmt)

    # Propagate phis backwards through the CFG
    def propagate_phi(block, name):
        for pred in block.preds:
            if name not in pred.exit_states:
                value = add_phi(pred, block.stmts, name, info=BI)
                propagate_phi(pred, name)

    for block in walk_blocks(first_block):
        for name in block.live_ins:
            propagate_phi(block, name)

    # Trim unneeded values from exit_states
    for block in walk_blocks(first_block):
        live_outs = {name: value for [name, value] in block.exit_states.items()
            if any(name in succ.live_ins for succ in block.succs)}
        for name in block.exit_states:
            if name not in live_outs:
                remove_use(Usage(block, 'exit_states', ArgType.DICT, index=name))
        block.exit_states = live_outs

    return first_block

################################################################################
## Optimization stuff ##########################################################
################################################################################

def can_dce(expr):
    return isinstance(expr, (syntax.BinaryOp, syntax.Integer, syntax.String,
            ExternSymbol))

def simplify_blocks(first_block):
    # Basic simplification pass. Right now, since we don't simplify
    # across blocks, this only needs to be run in one forward pass on
    # each block
    for block in walk_blocks(first_block):
        for stmt in block.stmts:
            # Simplify arithmetic expressions
            if (isinstance(stmt, syntax.BinaryOp) and
                    isinstance(stmt.lhs, syntax.Integer) and
                    isinstance(stmt.rhs, syntax.Integer) and
                    stmt.type not in {'and', 'or'}):
                [lhs, rhs] = [stmt.lhs.value, stmt.rhs.value]
                op = syntax.BINARY_OP_TABLE[stmt.type]
                result = getattr(lhs, op)(rhs)
                stmt.forward(syntax.Integer(result, info=stmt))
                # XXX this should not have to be done manually
                remove_use(Usage(stmt, 'lhs', ArgType.EDGE))
                remove_use(Usage(stmt, 'rhs', ArgType.EDGE))

            # Simplify intrinsic calls
            elif isinstance(stmt, syntax.Call):
                if isinstance(stmt.fn, Intrinsic):
                    stmt.fn.simplify_fn(stmt, stmt.args)

    # Run DCE
    any_removed = True
    while any_removed:
        any_removed = False
        for block in walk_blocks(first_block):
            statements = []
            for [i, stmt] in enumerate(block.stmts):
                if can_dce(stmt) and len(stmt._uses) == 1:
                    assert list(stmt._uses.values())[0].user == block
                    any_removed = True
                else:
                    statements.append(stmt)

                # XXX need to prevent deletion here eventually
                remove_use(Usage(block, 'stmts', ArgType.LIST, index=i))

            block.stmts = []
            for stmt in statements:
                append_to_edge_list(block, 'stmts', stmt)

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
            return [lir.cmp(node_map[node.lhs], node_map[node.rhs]), lir.Inst('set' + cc)]
    elif isinstance(node, Parameter):
        return lir.parameter(node.index)
    elif isinstance(node, ExternSymbol):
        return lir.literal(asm.ExternLabel(node.name))
    elif isinstance(node, syntax.Integer):
        return lir.literal(node.value)
    elif isinstance(node, syntax.Call):
        return lir.call(node_map[node.fn], *[node_map[arg] for arg in node.args])
    elif isinstance(node, syntax.Return):
        return lir.ret(node_map[node.expr])
    elif isinstance(node, Instruction):
        return lir.Inst(node.opcode, *[node_map[arg] for arg in node.args])
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
            assert stmt not in node_map
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

def compile_stmts(stmts):
    # Create a prelude with the standard C main parameters
    parameters = ['argc', 'argv']
    prelude = [syntax.Assignment(syntax.Target([name], info=BI), syntax.Parameter(i, info=BI), info=BI)
            for i, name in enumerate(parameters)]
    stmts = prelude + stmts

    ctx = syntax.Context('__main__', None, None)
    block = syntax.Block(stmts, info=BI)
    block = parse.preprocess_program(ctx, block, include_io_handlers=False)

    transform_to_graph(block)

    first_block = gen_ssa(block)

    simplify_blocks(first_block)

    lir_blocks = generate_lir(first_block)

    # Create a main function with all the code in it (no user-defined functions for now)
    fn = lir.Function('_main', parameters, lir_blocks)

    return [fn]

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
