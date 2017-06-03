import sys

import sprdpl.parse as libparse

import parse
from syntax import *

def add_to(cls):
    def deco(fn):
        setattr(cls, fn.__name__, fn)
        return fn
    return deco

# Machinery for 'smart' nodes in the IR graph, that track uses and can replace a given
# node in all nodes that use it.

# Edge class. This adds a layer of indirection, and is what is tracked by the node's
# 'refs' attribute. The indirect values/nodes are accessed by calling the edge, i.e.
# node.edge(), for brevity. Replacing the value is done by node.edge.set(value)
class Edge:
    def __init__(self, value):
        self.value = value
        if value is not None:
            value.add_ref(self)

    def __call__(self):
        return self.value

    def set(self, value):
        if self.value is not None:
            self.value.remove_ref(self)
        self.value = value
        if value is not None:
            value.add_ref(self)

@add_to(Node)
def add_ref(self, edge):
    assert edge not in self.refs
    self.refs.append(edge)

@add_to(Node)
def remove_ref(self, edge):
    assert edge in self.refs
    self.refs.remove(edge)

@add_to(Node)
def forward(self, new_value):
    for edge in self.refs:
        edge.set(new_value)

@add_to(Node)
def transform_to_graph(self, ctx):
    # First, add a list of references. This is used for value forwarding and (eventually) DCE.
    self.refs = []
    # Also add a stupid flag to make sure to only visit each node once. This is only
    # necessary because of the mg_builtins functions that get hackily imported, and
    # could possibly go away if that gets cleaned up.
    self.transformed = True

    # Recurse into the children nodes. This must be done before we transform our own references
    # so that the children's refs are set up.
    for node in self.iterate_children():
        if not hasattr(node, 'transformed'):
            node.transform_to_graph(ctx)

    # For every node that this node uses, replace the normal Python attribute with an Edge
    # containing the same node.
    for (arg_type, arg_name) in type(self).arg_defs:
        if arg_type in (ARG_EDGE, ARG_EDGE_OPT):
            node = getattr(self, arg_name)
            setattr(self, arg_name, Edge(node))
        elif arg_type == ARG_EDGE_LIST:
            nodes = getattr(self, arg_name)
            setattr(self, arg_name, [Edge(node) for node in nodes])
        elif arg_type == ARG_EDGE_DICT:
            nodes = getattr(self, arg_name)
            setattr(self, arg_name, {Edge(key): Edge(value) for key, value in nodes.items()})

def compile(path, print_program=False):
    ctx = Context('__main__', None, None)
    try:
        block = parse.parse(path, ctx=ctx)
    except libparse.ParseError as e:
        e.print_and_exit()
    preprocess_program(ctx, block)

    for expr in block:
        expr.transform_to_graph(ctx)

def main(args):
    compile(args[1])

if __name__ == '__main__':
    main(sys.argv)
