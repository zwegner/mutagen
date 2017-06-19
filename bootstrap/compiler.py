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

@add_to(syntax.Node)
def add_ref(self, edge):
    assert edge not in self.refs
    self.refs.append(edge)

@add_to(syntax.Node)
def remove_ref(self, edge):
    assert edge in self.refs
    self.refs.remove(edge)

@add_to(syntax.Node)
def forward(self, new_value):
    for edge in self.refs:
        edge.set(new_value)

def transform_to_graph(block):
    seen = set()
    for stmt in block:
        # Iterate the graph in reverse depth-first order to get a topological ordering
        for node in reversed(list(stmt.iterate_graph(seen))):
            # First, add a list of references. This is used for value forwarding and
            # (eventually) DCE.
            node.refs = []

            # For every node that this node uses, replace the normal Python attribute
            # with an Edge containing the same node.
            for (arg_type, arg_name) in type(node).arg_defs:
                if arg_type in (ArgType.EDGE, ArgType.OPT):
                    child = getattr(node, arg_name)
                    setattr(node, arg_name, Edge(child))
                elif arg_type == ArgType.LIST:
                    children = getattr(node, arg_name)
                    setattr(node, arg_name, [Edge(child) for child in children])
                elif arg_type == ArgType.DICT:
                    children = getattr(node, arg_name)
                    setattr(node, arg_name, collections.OrderedDict((Edge(key), Edge(value))
                        for key, value in children.items()))

def compile(path, print_program=False):
    ctx = syntax.Context('__main__', None, None)
    try:
        block = parse.parse(path, ctx=ctx)
    except libparse.ParseError as e:
        e.print_and_exit()
    syntax.preprocess_program(ctx, block)

    transform_to_graph(block)

def main(args):
    compile(args[1])

if __name__ == '__main__':
    main(sys.argv)
