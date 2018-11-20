import collections

import syntax

# This is a really basic shell of an implementation

# ABC for Inst/Phi for easy type checking
class Node:
    pass

class Phi(Node):
    def __init__(self, name, args):
        self.name = name
        self.args = args
    def __str__(self):
        return 'Phi({}, {})'.format(self.name, ', '.join(map(repr, self.args)))

class Inst(Node):
    def __init__(self, opcode, *args):
        self.opcode = opcode
        self.args = args
    def __str__(self):
        return '{}({})'.format(self.opcode, ', '.join(map(repr, self.args)))

class Function:
    def __init__(self, name, parameters, blocks):
        self.name = name
        self.parameters = parameters
        self.blocks = blocks

class BasicBlock:
    def __init__(self, name, phis, insts, preds, succs):
        self.name = name
        self.phis = phis
        self.insts = insts
        self.preds = preds
        self.succs = succs

def literal(a): return Inst('literal', a)

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

def phi_ref(name: str): return Inst('phi_ref', name)
