import collections

import syntax

# This is a really basic shell of an implementation

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
class NodeDict(collections.MutableMapping):
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

class Inst:
    def __init__(self, opcode, *args):
        self.opcode = opcode
        self.args = args
    def __repr__(self):
        return '{}({})'.format(self.opcode, ', '.join(map(repr, self.args)))

# Instruction wrappers
def test64(a, b): return Inst('test64', a, b)
def jz(a): return Inst('jz', a)
def jnz(a): return Inst('jnz', a)
def jmp(a): return Inst('jmp', a)
def ret(a): return Inst('ret', a)

def mov64(a): return Inst('mov64', a)
def add64(a, b): return Inst('add64', a, b)
def sub64(a, b): return Inst('sub64', a, b)
def mul64(a, b): return Inst('imul64', a, b)
def and64(a, b): return Inst('and64', a, b)
def or64(a, b): return Inst('or64', a, b)

def cmp64(a, b): return Inst('cmp64', a, b)

def call(fn, *args): return Inst('call', fn, *args)

def parameter(index): return Inst('parameter', index)
def phi(name: str, args): return Inst('phi', name, *args)
def phi_ref(name: str): return Inst('phi_ref', name)

# Node reference wrapper (NR for short)
class NR:
    def __init__(self, node):
        self.node = node
    def __repr__(self):
        return 'NR({})'.format(self.node)
