import struct

class Label(name: str, is_global: bool):
    def __str__(self):
        return '<{}>'.format(self.name)

class Relocation(label: Label, size: int):
    pass

class Register(index: int):
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
        else:
            if size == 8:
                if self.index & 4:
                    return names[self.index + 1] + 'l'
                return names[self.index + 1][0] + 'l'
            return prefix + names[self.index + 1]

class Address(base: int, scale: int, index: int, disp: int):
    def to_str(self, use_size_prefix, size):
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

def rex(w, r, addr):
    [r, x, b] = ex_transform(r, addr)
    value = w << 3 | (r & 8) >> 1 | (x & 8) >> 2 | (b & 8) >> 3
    if value:
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

class Instruction(opcode: str, size: int, *args):
    arg0_table = {
        'ret': [0xC3],
        'nop': [0x90],
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

    jump_table = {'j' + cond: 0x80 | code for [cond, code] in cond_table}
    setcc_table = {'set' + cond: 0x90 | code for [cond, code] in cond_table}

    cond_canon = {cond: conds[0] for conds in all_conds for cond in conds}
    canon_table = {prefix + cond: prefix + canon for prefix in ['j', 'set']
            for [cond, canon] in cond_canon}

    destructive_ops = set(arg2_table.keys() + shift_table.keys() +
            setcc_table.keys() + ['lea', 'pop'])

    def normalize_opcode(opcode):
        opcode = opcode.replace('8', '').replace('16', '')
        opcode = opcode.replace('32', '').replace('64', '')

        # Canonicalize instruction names that have multiple names
        if opcode in canon_table:
            opcode = canon_table[opcode]

        return opcode

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

        opcode = normalize_opcode(opcode)

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
            # XXX HACK: ah/bh etc. are used instead of sil etc. unless there's a REX
            if isinstance(dst, Register) and dst.index & 0xC == 4:
                prefix = [0x40]
            else:
                prefix = rex(0, 0, dst)
            return prefix + [0x0F, opcode] + mod_rm_sib(0, dst)
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

def is_destructive(opcode):
    opcode = Instruction.normalize_opcode(opcode)
    return opcode in Instruction.destructive_ops

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
            for byte in inst.to_bytes():
                if isinstance(byte, Relocation):
                    relocations = relocations + [[byte, len(bytes)]]
                    bytes = bytes + [0] * byte.size
                else:
                    bytes = bytes + [byte]

    # Fill in relocations
    if relocations:
        label_dict = dict(labels)
        new_bytes = []
        last_offset = 0
        for [rel, offset] in relocations:
            new_bytes = new_bytes + slice(bytes, last_offset, offset)
            # XXX only 4-byte for now
            assert rel.size == 4
            disp = label_dict[rel.label.name] - offset - rel.size
            new_bytes = new_bytes + pack32(disp)
            last_offset = offset + rel.size
        bytes = new_bytes + slice(bytes, last_offset, None)

    return [bytes, labels, global_labels]

# Create a big list of possible instructions with possible operands
inst_specs = []

for inst in Instruction.arg0_table.keys():
    inst_specs = inst_specs + [[inst]]

for size in [32, 64]:
    for inst in Instruction.arg1_table.keys():
        inst_specs = inst_specs + [['{}{}'.format(inst, size), 'ra']]

    for inst in Instruction.arg2_table.keys():
        inst_specs = inst_specs + [['{}{}'.format(inst, size), 'r', 'rai'],
                ['{}{}'.format(inst, size), 'a', 'r']]

    for inst in Instruction.shift_table.keys():
        inst_specs = inst_specs + [['{}{}'.format(inst, size), 'ra', 'i'],
            # Printing shifts by CL is a pain in the ass, since it can
            # use two different register sizes
            # ['{}{}'.format(inst, size), 'ra', Register(1)]
        ]

    for inst in Instruction.bmi_arg2_table.keys():
        inst_specs = inst_specs + [['{}{}'.format(inst, size), 'r', 'ra']]

    for inst in Instruction.bmi_arg3_table.keys():
        [src1, src2] = ['ra', 'r']
        if inst in Instruction.bmi_arg3_reversed:
            [src2, src1] = [src1, src2]
        inst_specs = inst_specs + [['{}{}'.format(inst, size), 'r', src1, src2]]

    inst_specs = inst_specs + [['push'.format(size), 'ri']]
    inst_specs = inst_specs + [['pop'.format(size), 'r']]
    inst_specs = inst_specs + [['lea{}'.format(size), 'r', 'a']]
    inst_specs = inst_specs + [['test{}'.format(size), 'ra', 'r']]

# Make sure all of the condition codes are tested, to test canonicalization
for [cond, code] in Instruction.jump_table:
    inst_specs = inst_specs + [[cond, 'l']]

for [cond, code] in Instruction.setcc_table:
    inst_specs = inst_specs + [[cond + '8', 'ra']]

for inst in ['jmp', 'call']:
    inst_specs = inst_specs + [[inst, 'rl']]
