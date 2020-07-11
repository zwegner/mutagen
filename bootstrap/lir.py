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
    def __init__(self, name, phis, insts, test, preds, succs):
        self.name = name
        self.phis = phis
        self.insts = insts
        self.test = test
        self.preds = preds
        self.succs = succs

def literal(a): return Inst('literal', a)

# Instruction wrappers
def test(a, b): return Inst('test', a, b)
def jz(a): return Inst('jz', a)
def jnz(a): return Inst('jnz', a)
def jmp(a): return Inst('jmp', a)
def ret(a): return Inst('ret', a)

def mov(a): return Inst('mov', a)
def add(a, b): return Inst('add', a, b)
def sub(a, b): return Inst('sub', a, b)
def mul(a, b): return Inst('imul', a, b)
def band(a, b): return Inst('and', a, b)
def bor(a, b): return Inst('or', a, b)

def cmp(a, b): return Inst('cmp', a, b)

def call(fn, *args): return Inst('call', fn, *args)

def parameter(index): return Inst('parameter', index)

def phi_ref(name: str): return Inst('phi_ref', name)
