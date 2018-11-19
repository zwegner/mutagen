#!/usr/bin/env python3
import collections
import sys

import sprdpl.parse as libparse

import parse
import syntax
from syntax import ArgType

def add_to(cls):
    def deco(fn):
        setattr(cls, fn.__name__, fn)
        return fn
    return deco

# Machinery for 'smart' nodes in the IR graph, that track uses and can replace a given
# node in all nodes that use it.

class Usage:
    def __init__(self, user, edge_name, type, index=None):
        self.user = user
        self.type = type
        self.edge_name = edge_name
        self.index = index
    def replace_use(self, old_node, new_node):
        if self.type in {ArgType.EDGE, ArgType.OPT}:
            assert getattr(self.user, self.edge_name) is old_node
            setattr(self.user, self.edge_name, new_node)
        elif self.type in {ArgType.LIST, ArgType.DICT}:
            container = getattr(self.user, self.edge_name)
            assert container[self.index] is old_node
            container[self.index] = new_node

        if old_node is not None:
            old_node._uses.pop(old_node._uses.index(self))

        if new_node is not None:
            new_node._uses.append(self)

@add_to(syntax.Node)
def forward(self, new_value):
    while self._uses:
        self._uses[0].replace_use(self, new_value)

@add_to(syntax.Node)
def remove_uses_by(self, user):
    # How much usage could a usage user use if a usage user could use usage?
    self._uses = [usage for usage in self._uses if usage.user is not user]

def set_edge(node, name, value):
    old = getattr(node, name)
    if old is not None:
        old.remove_uses_by(node)
    setattr(node, name, value)
    if value is not None:
        # XXX ArgType.EDGE is not necessarily accurate, for now rely on identical handling
        # of EDGE/OPT for Usage, cuz I'm lazy
        value._uses.append(Usage(node, name, ArgType.EDGE))

def append_to_edge_list(node, name, item):
    edge_list = getattr(node, name)
    item._uses.append(Usage(node, name, ArgType.LIST, index=len(edge_list)))
    edge_list.append(item)

def set_edge_key(node, name, key, value):
    edge_dict = getattr(node, name)
    value._uses.append(Usage(node, name, ArgType.DICT, index=key))
    edge_dict[key] = value

def transform_to_graph(block):
    # Iterate the graph in reverse depth-first order to get a topological ordering
    for node in reversed(list(block.iterate_graph())):
        node._uses = []
        # For every node that this node uses, add a Usage object to its _uses list.
        for (arg_type, arg_name) in type(node).arg_defs:
            child = getattr(node, arg_name)
            if arg_type in {ArgType.EDGE, ArgType.OPT}:
                if child is not None:
                    child._uses.append(Usage(node, arg_name, arg_type))
            elif arg_type in {ArgType.LIST, ArgType.DICT}:
                children = enumerate(child) if arg_type == ArgType.LIST else child.items()
                for index, item in children:
                    item._uses.append(Usage(node, arg_name, arg_type, index=index))

################################################################################
## CFG stuff ###################################################################
################################################################################

BLOCK_ID = 0
@syntax.node('*stmts, *phis, *preds, ?test, *succs, #exit_states')
class BasicBlock(syntax.Node):
    def setup(self):
        global BLOCK_ID
        self.block_id = BLOCK_ID
        BLOCK_ID += 1

# Just a dumb helper because our @node() decorator doesn't support keyword args/defaults
def basic_block(stmts=None, phis=None, preds=None, test=None, succs=None, exit_states=None):
    return BasicBlock(stmts or [], phis or [], preds or [], test, succs or [],
            exit_states or {}, info=syntax.BUILTIN_INFO)

@syntax.node('name')
class Phi(syntax.Node):
    def repr(self, ctx):
        return '<Phi "%s">' % self.name

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
        if block.phis:
            print('  phis:')
            for phi in block.phis:
                print('    ', phi.repr(None))
        if block.stmts:
            print('  stmts:')
            for stmt in block.stmts:
                print('    ', stmt.repr(None))
        if block.test:
            print('  test', block.test.repr(None))

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

################################################################################
## SSA stuff ###################################################################
################################################################################

def gen_ssa_for_stmt(block, statements, stmt):
    for node in reversed(list(stmt.iterate_graph())):
        # Handle loads
        if isinstance(node, syntax.Identifier):
            if node.name in block.exit_states:
                value = block.exit_states[node.name]
            else:
                value = Phi(node.name, info=node)
                append_to_edge_list(block, 'phis', value)
                set_edge_key(block, 'exit_states', node.name, value)
            node.forward(value)
        # Handle stores
        elif isinstance(node, syntax.Assignment):
            for target in node.target.targets:
                # XXX destructuring assignment is more complicated, we would need to desugar it
                # to involve temporaries and indexing
                assert isinstance(target, str)
                set_edge_key(block, 'exit_states', target, node.rhs)
        elif isinstance(node, syntax.Target):
            pass
        else:
            statements.append(node)

def gen_ssa(block):
    first_block = basic_block()
    last = block.gen_blocks(first_block)

    for block in walk_blocks(first_block):
        statements = []
        for stmt in block.stmts:
            add_stmt = gen_ssa_for_stmt(block, statements, stmt)

            # XXX need to prevent deletion here eventually
            stmt.remove_uses_by(block)

        if block.test:
            gen_ssa_for_stmt(block, statements, block.test)

        block.stmts = []
        for stmt in statements:
            append_to_edge_list(block, 'stmts', stmt)

    # Propagate phis backwards through the CFG
    def propagate_phi(block, phi):
        for pred in block.preds:
            if phi.name not in pred.exit_states:
                value = Phi(phi.name, info=phi)
                append_to_edge_list(pred, 'phis', value)
                set_edge_key(pred, 'exit_states', phi.name, value)
                propagate_phi(pred, phi)

    for block in walk_blocks(first_block):
        for phi in block.phis:
            propagate_phi(block, phi)

    return first_block

DCE_WHITELIST = {syntax.BinaryOp, syntax.Integer}

def simplify_blocks(first_block):
    # Basic simplification pass
    for block in walk_blocks(first_block):
        for stmt in block.stmts:
            if isinstance(stmt, syntax.BinaryOp):
                if isinstance(stmt.lhs, syntax.Integer) and isinstance(stmt.rhs, syntax.Integer):
                    if stmt.type not in {'and', 'or'}:
                        [lhs, rhs] = [stmt.lhs.value, stmt.rhs.value]
                        op = syntax.BINARY_OP_TABLE[stmt.type]
                        result = getattr(lhs, op)(rhs)
                        stmt.forward(syntax.Integer(result, info=stmt))
                        # XXX this should not have to be done manually
                        stmt.lhs.remove_uses_by(stmt)
                        stmt.rhs.remove_uses_by(stmt)

    # Run DCE
    any_removed = True
    while any_removed:
        any_removed = False
        for block in walk_blocks(first_block):
            statements = []
            for stmt in block.stmts:
                if type(stmt) in DCE_WHITELIST and len(stmt._uses) == 1:
                    assert stmt._uses[0].user == block
                    any_removed = True
                else:
                    statements.append(stmt)

                # XXX need to prevent deletion here eventually
                stmt.remove_uses_by(block)

            block.stmts = []
            for stmt in statements:
                append_to_edge_list(block, 'stmts', stmt)

def compile(path, print_program=False):
    ctx = syntax.Context('__main__', None, None)
    try:
        stmts = parse.parse(path, import_builtins=False, eval_ctx=ctx)
    except libparse.ParseError as e:
        e.print()
        sys.exit(1)
    block = syntax.Block(stmts, info=syntax.BUILTIN_INFO)
    block = parse.preprocess_program(ctx, block, include_io_handlers=False)

    transform_to_graph(block)

    first_block = gen_ssa(block)

    simplify_blocks(first_block)

    print_blocks(first_block)

def main(args):
    compile(args[1])

if __name__ == '__main__':
    main(sys.argv)
