import collections

from . import syntax

# This is a really basic shell of an implementation

# ABC for Inst/Phi for easy type checking
class Node:
    def __repr__(self):
        return '<{} at {}>'.format(self.opcode, hex(id(self)))
    def flatten(self):
        # XXX need to handle phis with dict args?
        for arg in self.args:
            if isinstance(arg, Node):
                yield from arg.flatten()
        yield self

# Phis are split into read/write phases, as in the TΦ transform from section
# 2.3.1 of "Register Allocation for Programs in SSA Form", Sebastian Hack 2006.
# This properly expresses the parallel-copy nature of phis, while keeping
# the semantics a bit simpler than the magic of normal phis. Note that this
# representation is basically the same as the "function call" formulation of SSA
# (which itself is like CPS without the passing). Each predecessor block in the
# CFG ends with a PhiR that reads all the live outs of the block, and each
# successor block starts with a PhiW that writes the live ins. In a
# sea-of-nodes IR like ours, we can't easily handle writing multiple values at
# the same time, so a PhiW is followed by some number of PhiSelects, that just
# select a single variable from the PhiW.

class PhiR(Node):
    def __init__(self, args: dict):
        self.opcode = 'PhiR'
        self.args = args
        self.regs = None
    def __str__(self):
        return 'PhiR({})'.format(', '.join('%s=%r' % (k, v)
                for [k, v] in sorted(self.args.items())))
    def __iter__(self):
        yield from sorted(self.args.items())

class PhiW(Node):
    def __init__(self, phi_reads, args: dict):
        self.opcode = 'PhiW'
        self.phi_reads = phi_reads
        self.args = args
    def __str__(self):
        return 'PhiW([{}], {})'.format(', '.join(map(repr, self.phi_reads)),
                ', '.join(self.args))

class PhiSelect(Node):
    def __init__(self, phi_write, name):
        self.opcode = 'PhiSelect'
        self.phi_write = phi_write
        self.name = name
        self.args = [phi_write]
    def __repr__(self):
        return 'PhiSelect({}, {})'.format(hex(id(self.phi_write)), self.name)

class Inst(Node):
    def __init__(self, opcode, *args):
        self.opcode = opcode
        self.args = args
    def __repr__(self):
        return '{}({})'.format(self.opcode, ', '.join(map(repr, self.args)))

class Address(Inst):
    def __init__(self, base, scale, index, disp):
        self.opcode = 'address'
        self.args = [base, scale, index, disp]

# Returns don't directly generate ret instructions, but are instead pseudo-ops
# that use the $return_value special variable... kinda icky but it works, since
# we want to use all the juicy SSA goodness
class Return(Node):
    def __init__(self, *args):
        self.opcode = 'return'
        self.args = args

class Function:
    def __init__(self, name, parameters, blocks, attributes=None):
        self.name = name
        self.parameters = parameters
        self.blocks = blocks
        self.attributes = attributes or {}

class BasicBlock:
    def __init__(self, name, phi_write, phi_selects, insts, test, phi_read,
            preds, succs):
        self.name = name
        self.phi_write = phi_write
        self.phi_selects = phi_selects
        self.insts = insts
        self.test = test
        self.phi_read = phi_read
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
