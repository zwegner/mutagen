import struct

class Label:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return '<{}>'.format(self.name)
    def __repr__(self):
        return '{}({})'.format(type(self).__name__, repr(self.name))

class LocalLabel(Label): pass

class GlobalLabel(Label): pass

class ExternLabel(Label): pass

class Relocation:
    def __init__(self, label, size: int):
        self.label = label
        self.size = size

class Register:
    def __init__(self, index: int):
        self.index = index
    def to_str(self, size):
        names = ['ip', 'ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di']
        [prefix, suffix] = {
            8: ['', 'b'],
            16: ['', 'w'],
            32: ['e', 'd'],
            64: ['r', ''],
        }[size]
        if self.index >= 8:
            return 'r{}{}'.format(self.index, suffix)
        elif size == 8:
            if self.index & 4:
                return names[self.index + 1] + 'l'
            return names[self.index + 1][0] + 'l'
        return prefix + names[self.index + 1]

class Address:
    def __init__(self, base: int, scale: int, index: int, disp: int):
        self.base = base
        self.scale = scale
        self.index = index
        self.disp = disp

    def to_str(self, use_size_prefix: bool, size: int):
        if use_size_prefix:
            size_str = {8: 'BYTE', 16: 'WORD', 32: 'DWORD', 64: 'QWORD'}[size]
            size_str = size_str + ' PTR '
        else:
            size_str = ''
        parts = [Register(self.base).to_str(64)]
        if self.scale:
            parts = parts + ['{}*{}'.format(Register(self.index).to_str(64),
                self.scale)]
        if self.disp:
            parts = parts + [self.disp]
        return '{}[{}]'.format(size_str, '+'.join(map(str, parts)))
    def __repr__(self):
        return 'Address({}, {}, {}, {})'.format(self.base, self.scale, self.index, self.disp)

def fits_8bit(imm: int):
    return -128 <= imm and imm <= 127

def pack8(imm: int):
    return list(struct.pack('<b', imm))

def pack32(imm: int):
    return list(struct.pack('<i', imm))

def pack64(imm: int):
    return list(struct.pack('<q', imm))

def mod_rm_sib(reg, rm):
    if isinstance(reg, Register):
        reg = reg.index
    sib_bytes = []
    disp_bytes = []
    if isinstance(rm, Register):
        mod = 3
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
            mod = 1
            disp_bytes = pack8(addr.disp)
        else:
            mod = 2
            disp_bytes = pack32(addr.disp)
    return [mod << 6 | (reg & 7) << 3 | base & 7] + sib_bytes + disp_bytes

def ex_transform(r, addr):
    if isinstance(addr, Address):
        [x, b] = [addr.index, addr.base]
    else:
        [x, b] = [0, addr]

    if isinstance(r, Register):
        r = r.index
    if isinstance(x, Register):
        x = x.index
    if isinstance(b, Register):
        b = b.index

    return [r, x, b]

def rex(w, r, addr, force=0):
    [r, x, b] = ex_transform(r, addr)
    value = w << 3 | (r & 8) >> 1 | (x & 8) >> 2 | (b & 8) >> 3
    if value or force:
        return [0x40 | value]
    return []

def vex(w, r, addr, m, v, l, p):
    [r, x, b] = ex_transform(r, addr)
    if isinstance(v, Register):
        v = v.index
    base = (~v & 15) << 3 | l << 2 | p
    if w or r or x or m != 1:
        return [0xC4, (~r & 8) << 4 | (~x & 8) << 3 | (~b & 8) << 2 | m,
                w << 7 | base]
    return [0xC4, (~r & 8) << 4 | base]

arg0_table = {
    'ret': [0xC3],
    'nop': [0x90],
}
arg1_table = {
    'not': 2,
    'neg': 3,
    'mul': 4,
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
shift_table = {
    'rol': 0,
    'ror': 1,
    'rcl': 2,
    'rcr': 3,
    'shl': 4,
    'shr': 5,
    'sar': 7,
}

bmi_arg2_table = {
    'lzcnt': 0xBD,
    'popcnt': 0xB8,
    'tzcnt': 0xBC,
}
bmi_arg3_table = {
    'andn':  [2, 0, 0xF2],
    'bextr': [2, 0, 0xF7],
    'bzhi':  [2, 0, 0xF5],
    'mulx':  [2, 3, 0xF6],
    'pdep':  [2, 3, 0xF5],
    'pext':  [2, 2, 0xF5],
    'sarx':  [2, 2, 0xF7],
    'shlx':  [2, 1, 0xF7],
    'shrx':  [2, 3, 0xF7],
}
bmi_arg3_reversed = ['andn', 'mulx', 'pdep', 'pext']

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

jump_table = {'j' + cond: 0x80 | code for [cond, code] in cond_table.items()}
setcc_table = {'set' + cond: 0x90 | code for [cond, code] in cond_table.items()}

cond_canon = {cond: conds[0] for conds in all_conds for cond in conds}
canon_table = {prefix + cond: prefix + canon for [cond, canon] in cond_canon.items()
        for prefix in ['j', 'set']}

destructive_ops = set(list(arg2_table.keys()) + list(shift_table.keys()) +
        list(setcc_table.keys()) + ['imul', 'lea', 'pop']) - {'cmp'}
no_reg_ops = {'cmp', 'test'}

jump_ops = set(list(jump_table.keys()) + ['jmp'])

def normalize_opcode(opcode):
    opcode = opcode.replace('8', '').replace('16', '')
    opcode = opcode.replace('32', '').replace('64', '')

    # Canonicalize instruction names that have multiple names
    if opcode in canon_table:
        opcode = canon_table[opcode]

    return opcode

class Instruction:
    def __init__(self, opcode: str, *args):
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

        self.opcode = normalize_opcode(opcode)
        self.size = size
        self.args = args

    def to_bytes(self):
        w = int(self.size == 64)

        if self.opcode in arg0_table:
            assert not self.args
            return arg0_table[self.opcode]
        elif self.opcode in arg1_table:
            opcode = arg1_table[self.opcode]
            [src] = self.args
            assert isinstance(src, Register) or isinstance(src, Address)
            return rex(w, 0, src) + [0xF7] + mod_rm_sib(opcode, src)
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
                    return rex(w, 0, dst) + [0xB8 | dst.index & 7] + imm_bytes
                else:
                    if fits_8bit(src):
                        [size_flag, imm_bytes] = [0x2, pack8(src)]
                    else:
                        [size_flag, imm_bytes] = [0, pack32(src)]
                    return rex(w, 0, dst) + [0x81 | size_flag] + mod_rm_sib(
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

            assert isinstance(src, Register)
            return rex(w, src, dst) + [opcode] + mod_rm_sib(src, dst)
        elif self.opcode in shift_table:
            sub_opcode = shift_table[self.opcode]
            [dst, src] = self.args
            suffix = []
            if isinstance(src, int):
                if src == 1:
                    opcode = 0xD1
                else:
                    opcode = 0xC1
                    suffix = pack8(src & 63)
            else:
                assert isinstance(src, Register)
                # Only CL
                assert src.index == 1
                opcode = 0xD3
            return rex(w, 0, dst) + [opcode] + mod_rm_sib(sub_opcode, dst) + suffix
        elif self.opcode == 'push':
            [src] = self.args
            if isinstance(src, int):
                if fits_8bit(src):
                    return [0x6A] + pack8(src)
                return [0x68] + pack32(src)
            else:
                assert isinstance(src, Register)
                return rex(0, 0, src) + [0x50 | (src.index & 7)]
        elif self.opcode == 'pop':
            [dst] = self.args
            assert isinstance(dst, Register)
            return rex(0, 0, dst) + [0x58 | (dst.index & 7)]
        elif self.opcode == 'imul':
            # XXX only 2-operand version for now
            [dst, src] = self.args
            assert isinstance(src, Register) or isinstance(src, Address)
            assert isinstance(dst, Register)
            return rex(w, dst, src) + [0x0F, 0xAF] + mod_rm_sib(dst, src)
        elif self.opcode == 'xchg':
            [dst, src] = self.args
            assert isinstance(src, Register)
            assert isinstance(dst, Register) or isinstance(dst, Address)
            return rex(w, src, dst) + [0x87] + mod_rm_sib(src, dst)
        elif self.opcode == 'lea':
            [dst, src] = self.args
            assert isinstance(dst, Register) and isinstance(src, Address)
            return rex(w, dst, src) + [0x8D] + mod_rm_sib(dst, src)
        elif self.opcode == 'test':
            # Test has backwards arguments, weird
            [src2, src1] = self.args
            return rex(w, src1, src2) + [0x85] + mod_rm_sib(src1, src2)
        elif self.opcode in bmi_arg2_table:
            opcode = bmi_arg2_table[self.opcode]
            [dst, src] = self.args
            return [0xF3] + rex(w, dst, src) + [0x0F, opcode] + mod_rm_sib(dst, src)
        elif self.opcode in bmi_arg3_table:
            [m, p, opcode] = bmi_arg3_table[self.opcode]
            [dst, src1, src2] = self.args
            if self.opcode in bmi_arg3_reversed:
                [src2, src1] = [src1, src2]
            return vex(w, dst, src1, m, src2, 0, p) + [opcode] + mod_rm_sib(dst, src1)
        elif self.opcode in jump_table:
            opcode = jump_table[self.opcode]
            [src] = self.args
            assert isinstance(src, Label)
            # XXX Since we don't know how far or in what direction we're jumping,
            # punt and use disp32. We'll fill the offset in later.
            return [0x0F, opcode, Relocation(src, 4)]
        elif self.opcode in ['jmp', 'call']:
            [src] = self.args
            [opcode, sub_opcode] = {'jmp': [0xE9, 4], 'call': [0xE8, 2]}[self.opcode]
            if isinstance(src, Label):
                return [opcode, Relocation(src, 4)]
            else:
                return rex(0, 0, src) + [0xFF] + mod_rm_sib(sub_opcode, src)
        elif self.opcode in setcc_table:
            opcode = setcc_table[self.opcode]
            [dst] = self.args
            # Force REX since ah/bh etc. are used instead of sil etc. without it
            force = isinstance(dst, Register) and dst.index & 0xC == 4
            return rex(0, 0, dst, force=force) + [0x0F, opcode] + mod_rm_sib(0, dst)
        assert False

    def __str__(self):
        if self.args:
            args = []
            for arg in self.args:
                if isinstance(arg, Register):
                    args = args + [arg.to_str(self.size)]
                elif isinstance(arg, Address):
                    # lea doesn't need a size since it's only the address...
                    use_size_prefix = self.opcode != 'lea'
                    args = args + [arg.to_str(use_size_prefix, self.size)]
                else:
                    if self.opcode in shift_table:
                        arg = arg & 63
                    # XXX handle different immediate sizes (to convert to unsigned)
                    args = args + [str(arg)]
            argstr = ' ' + ','.join(args)
        else:
            argstr = ''
        return self.opcode + argstr

def is_destructive_op(opcode):
    opcode = normalize_opcode(opcode)
    return opcode in destructive_ops

def needs_register(opcode):
    opcode = normalize_opcode(opcode)
    return opcode not in no_reg_ops

def is_jump_op(opcode):
    opcode = normalize_opcode(opcode)
    return opcode in jump_ops

def build(insts):
    bytes = []
    local_labels = global_labels = extern_labels = []
    relocations = []
    for inst in insts:
        if isinstance(inst, LocalLabel):
            local_labels = local_labels + [[inst.name, len(bytes)]]
        elif isinstance(inst, GlobalLabel):
            global_labels = global_labels + [[inst.name, len(bytes)]]
        else:
            for byte in inst.to_bytes():
                if isinstance(byte, Relocation):
                    # If the relocation is to a an external symbol, pass it on
                    if isinstance(byte.label, ExternLabel):
                        assert byte.size == 4
                        extern_labels = extern_labels + [[byte.label.name, len(bytes)]]
                        # HACKish: assume the relocation is at the end of an instruction.
                        # Since the PC will be 4 bytes after the end of this value when the
                        # instruction executes, and the linker will calculate the offset from
                        # the beginning of the value, put an offset of -4 here that the linker
                        # will add in.
                        bytes = bytes + pack32(-4)
                    else:
                        relocations = relocations + [[byte, len(bytes)]]
                        bytes = bytes + [0] * byte.size
                else:
                    bytes = bytes + [byte]

    # Fill in relocations
    if relocations:
        label_dict = dict(local_labels + global_labels)
        new_bytes = []
        last_offset = 0
        for [rel, offset] in relocations:
            new_bytes = new_bytes + bytes[last_offset:offset]
            # XXX only 4-byte for now
            assert rel.size == 4
            disp = label_dict[rel.label.name] - offset - rel.size
            new_bytes = new_bytes + pack32(disp)
            last_offset = offset + rel.size
        bytes = new_bytes + bytes[last_offset:]

    return [bytes, local_labels, global_labels, extern_labels]

# Create a big list of possible instructions with possible operands
def get_inst_specs():
    for inst in arg0_table.keys():
        yield [inst]

    for size in [32, 64]:
        for inst in arg1_table.keys():
            yield ['{}{}'.format(inst, size), 'ra']

        for inst in arg2_table.keys():
            yield ['{}{}'.format(inst, size), 'r', 'rai']
            yield ['{}{}'.format(inst, size), 'a', 'r']

        for inst in shift_table.keys():
            yield ['{}{}'.format(inst, size), 'ra', 'i']
            # Printing shifts by CL is a pain in the ass, since it can
            # use two different register sizes
            # yield ['{}{}'.format(inst, size), 'ra', Register(1)]

        for inst in bmi_arg2_table.keys():
            yield ['{}{}'.format(inst, size), 'r', 'ra']

        for inst in bmi_arg3_table.keys():
            [src1, src2] = ['ra', 'r']
            if inst in bmi_arg3_reversed:
                [src2, src1] = [src1, src2]
            yield ['{}{}'.format(inst, size), 'r', src1, src2]

        # Push/pop don't have a 32-bit operand encoding in 64-bit mode...
        yield ['push', 'ri']
        yield ['pop', 'r']

        yield ['imul{}'.format(size), 'r', 'ra']
        yield ['xchg{}'.format(size), 'ra', 'r']
        yield ['lea{}'.format(size), 'r', 'a']
        yield ['test{}'.format(size), 'ra', 'r']

    # Make sure all of the condition codes are tested, to test canonicalization
    for [cond, code] in jump_table.items():
        yield [cond, 'l']

    for [cond, code] in setcc_table.items():
        yield [cond + '8', 'ra']

    for inst in ['jmp', 'call']:
        yield [inst, 'rl']
