import struct

import elf

class Label(name: str, is_global: bool):
    def __str__(self):
        return '<{}>'.format(self.name)

# XXX includes the bytes of the full instruction, as it's slightly more
# convenient. Not sure how to do this more cleanly...
class Relocation(label: Label, code_offset: int, size: int, bytes):
    pass

class Register(index: int):
    def to_str(self, size):
        names = ['ip', 'ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di']
        [prefix, suffix] = {
            8: ['', 'l'],
            16: ['', 'w'],
            32: ['e', 'd'],
            64: ['r', ''],
        }[size]
        if self.index >= 8:
            return 'r{}{}'.format(self.index, suffix)
        else:
            if size == 8:
                if self.index & 4:
                    return names[self.index + 1] + 'l'
                return names[self.index + 1][0] + 'l'
            return prefix + names[self.index + 1]

class Address(base, scale, index, disp):
    def to_str(self, size):
        size_str = {8: 'BYTE', 16: 'WORD', 32: 'DWORD', 64: 'QWORD'}[size]
        parts = [Register(self.base).to_str(64)]
        if self.scale:
            parts = parts + ['{}*{}'.format(Register(self.index).to_str(64),
                self.scale)]
        if self.disp:
            parts = parts + [self.disp]
        return '{} PTR [{}]'.format(size_str, '+'.join(map(str, parts)))

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

class Instruction(opcode: str, size: int, *args):
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

    all_conds = [
        ['o'],
        ['no'],
        ['b', 'c', 'nae'],
        ['ae', 'nb', 'nc'],
        ['e', 'z'],
        ['ne', 'nz'],
        ['be', 'na'],
        ['a', 'nbe'],
        ['s'],
        ['ns'],
        ['p', 'pe'],
        ['np', 'po'],
        ['l', 'nge'],
        ['ge', 'nl'],
        ['le', 'ng'],
        ['g', 'nle'],
    ]
    cond_table = {cond: i for [i, conds] in enumerate(all_conds) for cond in conds}

    jump_table = {'j' + cond: 0x80 | code for [cond, code] in cond_table}
    setcc_table = {'set' + cond: 0x90 | code for [cond, code] in cond_table}

    cond_canon = {cond: conds[0] for conds in all_conds for cond in conds}
    canon_table = {prefix + cond: prefix + canon for prefix in ['j', 'set']
            for [cond, canon] in cond_canon}

    def __init__(opcode: str, *args):
        # Handle 32/64 bit instruction size. This info is stuck in the opcode
        # name for now since not all instructions need it.
        if opcode.endswith('8'):
            size = 8
        elif opcode.endswith('16'):
            size = 16
        elif opcode.endswith('32'):
            size = 32
        elif opcode.endswith('64'):
            size = 64
        else:
            # XXX default size--this might need more logic later
            size = 64
        opcode = opcode.replace('8', '').replace('16', '')
        opcode = opcode.replace('32', '').replace('64', '')

        # Canonicalize instruction names that have multiple names
        if opcode in canon_table:
            opcode = canon_table[opcode]

        # This dictionary shit really needs to go. Need polymorphism!
        return {'opcode': opcode, 'size': size, 'args': args}

    def to_bytes(self):
        w = int(self.size == 64)

        if self.opcode in arg0_table:
            assert not self.args
            return arg0_table[self.opcode]
        elif self.opcode in arg1_table:
            opcode = arg1_table[self.opcode]
            [src] = self.args
            assert isinstance(src, Register) or isinstance(src, Address)
            return rex(w, 0, 0, src.index) + [0xF7] + mod_rm_sib(opcode, src)
        elif self.opcode in arg2_table:
            opcode = arg2_table[self.opcode]
            [dst, src] = self.args

            # Immediates have separate opcodes, so handle them specially here.
            if isinstance(src, int):
                assert isinstance(dst, Register)
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
                # op reg, reg
                assert isinstance(src, Register)
                return rex(w, src, 0, dst) + [opcode] + mod_rm_sib(src, dst)
            else:
                assert isinstance(src, Register)
                return rex_addr(w, src, dst) + [opcode] + mod_rm_sib(src, dst)
        elif self.opcode in jump_table:
            opcode = jump_table[self.opcode]
            [dst] = self.args
            assert isinstance(dst, Label)
            # XXX Since we don't know how far or in what direction we're jumping,
            # punt and use the 32-bit displacement and fill it with zeroes. We'll
            # fill all the offsets in later.
            bytes = [0x0F, opcode, 0, 0, 0, 0]
            return Relocation(dst, 2, 4, bytes)
        elif self.opcode in setcc_table:
            opcode = setcc_table[self.opcode]
            [dst] = self.args
            # XXX HACK: ah/bh etc. are used instead of sil etc. unless there's a REX
            if isinstance(dst, Register) and dst.index & 0xC == 4:
                prefix = [0x40]
            else:
                prefix = []
            return prefix + [0x0F, opcode] + mod_rm_sib(0, dst)
        assert False

    def __str__(self):
        if self.args:
            args = []
            for arg in self.args:
                if isinstance(arg, Register) or isinstance(arg, Address):
                    args = args + [arg.to_str(self.size)]
                else:
                    # XXX handle different immediate sizes (to convert to unsigned)
                    args = args + [str(arg)]
            argstr = ' ' + ','.join(args)
        else:
            argstr = ''
        return self.opcode + argstr

def build(insts):
    bytes = []
    labels = []
    global_labels = []
    relocations = []
    for inst in insts:
        if isinstance(inst, Label):
            labels = labels + [[inst.name, len(bytes)]]
            if inst.is_global:
                global_labels = global_labels + [inst.name]
        else:
            b = inst.to_bytes()
            if isinstance(b, Relocation):
                relocations = relocations + [[b, len(bytes) + b.code_offset]]
                b = b.bytes
            bytes = bytes + b

    # Fill in relocations
    if relocations:
        # XXX this is basically the dict() constructor
        label_dict = {name: offset for [name, offset] in labels}
        new_bytes = []
        last_offset = 0
        for [rel, offset] in relocations:
            new_bytes = new_bytes + slice(bytes, last_offset, offset)
            # XXX only 4-byte for now
            assert rel.size == 4
            disp = label_dict[rel.label.name] - offset - rel.size
            new_bytes = new_bytes + pack32(disp)
            last_offset = offset + rel.size
        bytes = new_bytes + slice(bytes, last_offset, len(bytes))

    return [bytes, labels, global_labels]

# Just a bunch of random instructions to test out different encodings.
insts = [
    Label('_test', True),
    Instruction('xor32', Register(0), Register(0)),
    Instruction('add32', Register(0), Register(1)),
    Instruction('cmp32', Register(0), Register(12)),
    Instruction('add32', Address(3, 8, 3, 0xFFFF), Register(1)),
    Instruction('add32', Register(1), Address(-1, 0, 0, 0xFFFF)),
    Instruction('add32', Address(-1, 0, 0, 0xFFFF), Register(1)),
    Instruction('add32', Address(3, 8, 3, 0), Register(1)),
    Instruction('add32', Address(5, 8, 5, 0xFFFF), Register(1)),
    Instruction('add32', Address(5, 8, 5, 0), Register(1)),
    Instruction('mov32', Address(3, 8, 3, 0xFFFF), Register(1)),
    Instruction('mov32', Register(1), Address(-1, 0, 0, 0xFFFF)),
    Instruction('mov32', Address(-1, 0, 0, 0xFFFF), Register(1)),
    # Test backward/forward jumps
    Instruction('jz', Label('_test', True)),
    Instruction('ja', Label('_test2', True)),
    Instruction('mov32', Address(3, 8, 3, 0), Register(1)),
    Instruction('mov32', Address(5, 8, 5, 0xFFFF), Register(1)),
    Instruction('mov32', Address(5, 8, 5, 0), Register(1)),
    Instruction('mov32', Register(1), Register(14)),
    Instruction('mov32', Register(1), Register(7)),
    Instruction('mov64', Register(1), Register(14)),
    Instruction('add32', Register(1), 4),
    Label('_test2', True),
    # Our IR can't properly print unsigned integers without a bunch of work,
    # as they appear in the objdump output. So ignore for now.
    #Instruction('add32', Register(1), -2),
    #Instruction('add32', Register(1), -0x200),
    Instruction('add32', Register(1), 0xFFF),
    Instruction('not32', Register(1)),
    Instruction('neg32', Register(1)),
    Instruction('mul32', Register(1)),
    Instruction('imul32', Register(1)),
    Instruction('div32', Register(1)),
    Instruction('idiv32', Register(1)),
    Instruction('not32', Address(5, 8, 5, 0xFFFF)),
    Instruction('neg32', Address(5, 8, 5, 0xFFFF)),
    Instruction('mul32', Address(5, 8, 5, 0xFFFF)),
    Instruction('imul32', Address(5, 8, 5, 0xFFFF)),
    Instruction('div32', Address(5, 8, 5, 0xFFFF)),
    Instruction('idiv32', Address(5, 8, 5, 0xFFFF)),
    Instruction('mov64', Register(0), 0x7FFFF000DEADBEEF),
    Instruction('ret'),
    Label('_test3', True),
    Instruction('mov64', Register(0), 0x0123456789ABCDEF),
    Instruction('ret'),
    Label('_jump_test', True),
]
# Make sure all of the condition codes are tested, to test canonicalization
for [cond, code] in Instruction.jump_table:
    insts = insts + [Instruction(cond, Label('_jump_test', True))]
# Same for setcc r/m
for [cond, code] in Instruction.setcc_table:
    insts = insts + [Instruction(cond + '8', Register(6)),
            Instruction(cond + '8', Address(3, 4, 5, 0xFF))]

elf_file = elf.create_elf_file(*build(insts))
write_binary_file('elfout.o', elf_file)

# Print out our interpretation of the instructions, so it can be matched
# against objdump
for inst in insts:
    if isinstance(inst, Label):
        print('{}:'.format(inst.name))
    else:
        print('    {}'.format(inst))
