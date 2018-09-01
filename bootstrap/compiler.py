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
    for usage in self._uses:
        usage.replace_use(self, new_value)

@add_to(syntax.Node)
def remove_uses_by(self, user):
    self._uses = [usage for usage in self._uses if usage.user is not user]

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
    def setup(self):
        self._uses = []
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
    return None

@add_to(syntax.Block)
def gen_blocks(self, current):
    for stmt in self.stmts:
        result = stmt.gen_blocks(current)
        if result:
            current = result
        else:
            current.stmts.append(stmt)
    return current

@add_to(syntax.While)
def gen_blocks(self, current):
    test_block = basic_block(test=self.expr)
    first = basic_block()
    exit_block = basic_block()

    last = self.block.gen_blocks(first)
    test_block_last = self.expr.gen_blocks(test_block)
    assert test_block == test_block_last

    link_blocks(current, test_block)
    link_blocks(test_block, first)
    link_blocks(test_block, exit_block)
    link_blocks(last, test_block)
    return exit_block

@add_to(syntax.IfElse)
def gen_blocks(self, current):
    assert not current.test
    self.expr.gen_blocks(current)
    current.test = self.expr
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

def gen_ssa(block):
    first_block = basic_block()
    last = block.gen_blocks(first_block)

    for block in walk_blocks(first_block):
        statements = []
        for stmt in block.stmts:
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

            # XXX need to prevent deletion here eventually
            stmt.remove_uses_by(block)

        block.stmts = []
        for stmt in statements:
            append_to_edge_list(block, 'stmts', stmt)

    # Propagate phis backwards through the CFG
    def propagate_phi(block, phi_name):
        for pred in block.preds:
            if phi_name not in pred.exit_states:
                value = Phi(phi_name, info=node)
                append_to_edge_list(pred, 'phis', value)
                set_edge_key(pred, 'exit_states', phi_name, value)
                propagate_phi(pred, phi_name)

    for block in walk_blocks(first_block):
        for phi in block.phis:
            propagate_phi(block, phi.name)

    return first_block

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

    print_blocks(first_block)

def main(args):
    compile(args[1])

if __name__ == '__main__':
    main(sys.argv)
