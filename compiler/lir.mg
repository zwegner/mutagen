# This is a really basic shell of an implementation

class Inst(opcode, *args):
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
def and64(a, b): return Inst('and64', a, b)
def or64(a, b): return Inst('or64', a, b)
def call(fn, *args): return Inst('call', fn, *args)

def parameter(index): return Inst('parameter', index)
def phi(name: str, args): return Inst('phi', name, *args)
def phi_ref(name: str): return Inst('phi_ref', name)

# Node reference wrapper (NR for short)
class NR(node_id: int):
    def __repr__(self):
        return 'NR({})'.format(self.node_id)
