import struct

import elf

class Label(name: str, is_global: bool):
    pass

class Register(index, size):
    def __str__(self):
        names = ['ip', 'ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di']
        if self.size == 32:
            prefix = 'e'
            suffix = 'd'
        else:
            assert self.size == 64
            prefix = 'r'
            suffix = ''
        if self.index >= 8:
            return 'r' + str(self.index) + suffix
        else:
            return prefix + names[self.index + 1]

class Address(base, scale, index, disp):
    def __str__(self):
        parts = [str(Register(self.base, 64))]
        if self.scale:
            parts = parts + [str(Register(self.index, 64)) + '*' + str(self.scale)]
        if self.disp:
            parts = parts + [str(self.disp)]
        return 'DWORD PTR [' + '+'.join(parts) + ']'

def fits_8bit(imm):
    return -128 <= imm and imm <= 127

def pack8(imm):
    return struct.pack('<b', imm)

def pack32(imm):
    return struct.pack('<i', imm)

def pack64(imm):
    return struct.pack('<q', imm)

def mod_rm_sib(reg, rm):
    if isinstance(reg, Register):
        reg = reg.index
    sib_bytes = []
    disp_bytes = []
    if isinstance(rm, Register):
        mod = 0xC0
        base = rm.index
    else:
        addr = rm
        base = addr.base
        if addr.scale or addr.base & 7 == 4:
            assert addr.base != -1
            assert addr.index != 4
            base = 4
            scale = {1: 0, 2: 1, 4: 2, 8: 3}[addr.scale]
            sib_bytes = [scale << 6 | (addr.index & 7) << 3 | addr.base & 7]

        # RIP+offset
        if addr.base == -1:
            mod = 0
            base = 5
            disp_bytes = pack32(addr.disp)
        # Various disp sizes--base==5 is used for RIP+offset, so needs a disp
        elif not addr.disp and addr.base & 7 != 5:
            mod = 0
        elif fits_8bit(addr.disp):
            mod = 0x40
            disp_bytes = pack8(addr.disp)
        else:
            mod = 0x80
            disp_bytes = pack32(addr.disp)
    return [mod | (reg & 7) << 3 | base & 7] + sib_bytes + disp_bytes

def rex(w, r, x, b):
    if isinstance(r, Register):
        r = r.index
    if isinstance(x, Register):
        x = x.index
    if isinstance(b, Register):
        b = b.index
    value = w << 3 | (r & 8) >> 1 | (x & 8) >> 2 | (b & 8) >> 3
    if value:
        return [0x40 | value]
    return []

def rex_addr(w, r, addr):
    return rex(w, r, addr.index, addr.base)

class Instruction(opcode: str, *args):
    arg0_table = {
        'ret': [0xC3],
    }
    arg1_table = {
        'not': 2,
        'neg': 3,
        'mul': 4,
        'imul': 5,
        'div': 6,
        'idiv': 7,
    }
    arg2_table = {
        'mov': -1, # handled separately
        'add': 0,
        'or': 1,
        'adc': 2,
        'sbb': 3,
        'and': 4,
        'sub': 5,
        'xor': 6,
        'cmp': 7,
    }
    def to_bytes(self):
        if self.opcode in arg0_table:
            assert not self.args
            return arg0_table[self.opcode]
        elif self.opcode in arg1_table:
            opcode = arg1_table[self.opcode]
            [dst] = self.args
            if isinstance(dst, Register):
                w = int(dst.size == 64)
                return rex(w, 0, 0, dst.index) + [0xF7] + mod_rm_sib(opcode, dst)
            else:
                # Need to handle size of address somehow
                assert False
        elif self.opcode in arg2_table:
            opcode = arg2_table[self.opcode]
            [dst, src] = self.args

            # Immediates have separate opcodes, so handle them specially here.
            if isinstance(src, int):
                assert isinstance(dst, Register)
                w = int(dst.size == 64)
                if self.opcode == 'mov':
                    if w:
                        imm_bytes = pack64(src)
                    else:
                        imm_bytes = pack32(src)
                    return rex(w, 0, 0, dst) + [0xB8 | dst.index & 7] + imm_bytes
                else:
                    if fits_8bit(src):
                        [size_flag, imm_bytes] = [0x2, pack8(src)]
                    else:
                        [size_flag, imm_bytes] = [0, pack32(src)]
                    return rex(w, 0, 0, dst) + [0x81 | size_flag] + mod_rm_sib(
                            opcode, dst) + imm_bytes

            # Mov is also a bit different, but can mostly be handled like other ops
            if self.opcode == 'mov':
                opcode = 0x89
            else:
                opcode = 1 | opcode << 3

            # op reg, mem is handled by flipping the direction bit and
            # swapping src/dst.
            if isinstance(src, Address):
                opcode = opcode | 0x2
                [src, dst] = [dst, src]

            if isinstance(dst, Register):
                w = int(dst.size == 64)
                # op reg, reg
                assert isinstance(src, Register)
                assert dst.size == src.size
                return rex(w, src, 0, dst) + [opcode] + mod_rm_sib(src, dst)
            else:
                assert isinstance(src, Register)
                w = int(src.size == 64)
                return rex_addr(w, src, dst) + [opcode] + mod_rm_sib(src, dst)
    def __str__(self):
        return self.opcode + ' ' + ','.join(map(str, self.args))

def build(insts):
    bytes = []
    labels = []
    global_labels = []
    for inst in insts:
        if isinstance(inst, Label):
            labels = labels + [[inst.name, len(bytes)]]
            if inst.is_global:
                global_labels = global_labels + [inst.name]
        else:
            bytes = bytes + inst.to_bytes()
    return [bytes, labels, global_labels]

# Just a bunch of random instructions to test out different encodings.
insts = [
    Label('_test', True),
    Instruction('xor', Register(0, 32), Register(0, 32)),
    Instruction('add', Register(0, 32), Register(1, 32)),
    Instruction('cmp', Register(0, 32), Register(12, 32)),
    Instruction('add', Address(3, 8, 3, 0xFFFF), Register(1, 32)),
    Instruction('add', Register(1, 32), Address(-1, 0, 0, 0xFFFF)),
    Instruction('add', Address(-1, 0, 0, 0xFFFF), Register(1, 32)),
    Instruction('add', Address(3, 8, 3, 0), Register(1, 32)),
    Instruction('add', Address(5, 8, 5, 0xFFFF), Register(1, 32)),
    Instruction('add', Address(5, 8, 5, 0), Register(1, 32)),
    Instruction('mov', Address(3, 8, 3, 0xFFFF), Register(1, 32)),
    Instruction('mov', Register(1, 32), Address(-1, 0, 0, 0xFFFF)),
    Instruction('mov', Address(-1, 0, 0, 0xFFFF), Register(1, 32)),
    Instruction('mov', Address(3, 8, 3, 0), Register(1, 32)),
    Instruction('mov', Address(5, 8, 5, 0xFFFF), Register(1, 32)),
    Instruction('mov', Address(5, 8, 5, 0), Register(1, 32)),
    Instruction('mov', Register(1, 32), Register(14, 32)),
    Instruction('mov', Register(1, 32), Register(7, 32)),
    Instruction('mov', Register(1, 64), Register(14, 64)),
    Instruction('add', Register(1, 32), 4),
    # Our IR can't properly print unsigned integers without a bunch of work,
    # as they appear in the objdump output. So ignore for now.
    #        Instruction('add', Register(1, 32), -2),
    #        Instruction('add', Register(1, 32), -0x200),
    Instruction('add', Register(1, 32), 0xFFF),
    Instruction('not', Register(1, 32)),
    Instruction('neg', Register(1, 32)),
    Instruction('mul', Register(1, 32)),
    Instruction('imul', Register(1, 32)),
    Instruction('div', Register(1, 32)),
    Instruction('idiv', Register(1, 32)),
    Instruction('mov', Register(0, 64), 0x7ffff000deadbeef),
    Instruction('ret'),
    Label('_test2', True),
    Instruction('mov', Register(0, 64), 0x0123456789abcdef),
    Instruction('ret'),
]

elf_file = elf.create_elf_file(*build(insts))
write_binary_file('elfout.o', elf_file)

# Print out our interpretation of the instructions, so it can be matched
# against objdump
for inst in insts:
    if isinstance(inst, Label):
        print(inst.name + ':')
    else:
        print('    ' + str(inst))
