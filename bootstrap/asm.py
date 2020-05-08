import struct

# Dummy abstract base class
class ASMObj: pass

class Label(ASMObj):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return '<{}>'.format(self.name)
    def __repr__(self):
        return '{}({})'.format(type(self).__name__, repr(self.name))
    def get_size(self):
        return None

class LocalLabel(Label): pass

class GlobalLabel(Label): pass

class ExternLabel(Label): pass

class Relocation(ASMObj):
    def __init__(self, label, size: int):
        self.label = label
        self.size = size

class Immediate(ASMObj):
    def __init__(self, value: int, size: int=None):
        self.value = value
        self.size = size
    def get_size(self):
        return self.size
    def __str__(self):
        return str(self.value)

# General-purpose register
class GPReg(ASMObj):
    def __init__(self, index: int, size: int=None):
        self.index = index
        self.size = size or 64 # XXX default of 64

    def get_size(self):
        return self.size

    def __str__(self):
        names = ['ip', 'ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di']
        [prefix, suffix] = {
            8: ['', 'b'],
            16: ['', 'w'],
            32: ['e', 'd'],
            64: ['r', ''],
        }[self.size]
        if self.index >= 8:
            return 'r{}{}'.format(self.index, suffix)
        elif self.size == 8:
            if self.index & 4:
                return names[self.index + 1] + 'l'
            return names[self.index + 1][0] + 'l'
        return prefix + names[self.index + 1]

class Address(ASMObj):
    def __init__(self, base: int, scale: int, index: int, disp: int, size: int=None):
        self.base = base
        self.scale = scale
        self.index = index
        self.disp = disp
        self.size = size

    def get_size(self):
        return self.size

    def to_str(self, use_size_prefix: bool, size: int):
        if use_size_prefix:
            size_str = {8: 'BYTE', 16: 'WORD', 32: 'DWORD', 64: 'QWORD'}[size]
            size_str = size_str + ' PTR '
        else:
            size_str = ''
        # XXX this SIB stuff assumes GPR, doesn't handle gather/scatter
        parts = [GPReg(self.base, size=64)]
        if self.scale:
            parts = parts + ['{}*{}'.format(GPReg(self.index, size=64),
                self.scale)]
        if self.disp:
            parts = parts + [self.disp]
        return '{}[{}]'.format(size_str, '+'.join(map(str, parts)))
    def __repr__(self):
        return 'Address({}, {}, {}, {})'.format(self.base, self.scale, self.index, self.disp)

def fits_8bit(imm: int):
    limit = 1 << 7
    return -limit <= imm <= limit - 1

def fits_32bit(imm: int):
    limit = 1 << 31
    return -limit <= imm <= limit - 1

def pack8(imm: int):
    return list(struct.pack('<b', imm))

def pack32(imm: int):
    return list(struct.pack('<i', imm))

def pack64(imm: int):
    return list(struct.pack('<q', imm))

# Basic helper function to factor out a common pattern of choosing a 1/4 byte encoding,
# along with another value that depends on which is chosen
def choose_8_or_32_bit(imm, op8, op32):
    if fits_8bit(imm):
        return [pack8(imm), op8]
    assert fits_32bit(imm)
    return [pack32(imm), op32]

def mod_rm_sib(reg, rm):
    if isinstance(reg, GPReg):
        reg = reg.index
    sib_bytes = []
    disp_bytes = []
    if isinstance(rm, GPReg):
        mod = 3
        base = rm.index
    elif isinstance(rm, Label):
        mod = 0
        base = 5
        disp_bytes = [Relocation(rm, 4)]
    else:
        addr = rm
        base = addr.base
        # RSP/R12 base needs a SIB byte. It uses RSP as an index, and scale is ignored
        if addr.base & 7 == 4 and not addr.scale:
            sib_bytes = [4 << 3 | 4]
        elif addr.scale:
            assert addr.base != -1
            assert addr.index != 4
            base = 4
            scale = {1: 0, 2: 1, 4: 2, 8: 3}[addr.scale]
            sib_bytes = [scale << 6 | (addr.index & 7) << 3 | addr.base & 7]

        # RIP+offset: this steals the encoding for RBP with no displacement
        if addr.base == -1:
            mod = 0
            base = 5
            disp_bytes = pack32(addr.disp)
        # Otherwise, encode the displacement as 1 or 4 bytes if needed: if the disp is nonzero
        # obviously, but also if RBP is the base, the no-disp encoding is stolen above, so
        # encode that with a single byte displacement of zero.
        elif addr.disp or addr.base & 7 == 5:
            [disp_bytes, mod] = choose_8_or_32_bit(addr.disp, 1, 2)
        else:
            mod = 0
    return [mod << 6 | (reg & 7) << 3 | base & 7] + sib_bytes + disp_bytes

def ex_transform(r, addr):
    if isinstance(addr, Address):
        [x, b] = [addr.index, addr.base]
    elif isinstance(addr, Label):
        [x, b] = [0, 5]
    else:
        [x, b] = [0, addr]

    if isinstance(r, GPReg):
        r = r.index
    if isinstance(x, GPReg):
        x = x.index
    if isinstance(b, GPReg):
        b = b.index

    return [r, x, b]

def rex(w, r, addr, force=False):
    [r, x, b] = ex_transform(r, addr)
    value = w << 3 | (r & 8) >> 1 | (x & 8) >> 2 | (b & 8) >> 3
    if value or force:
        return [0x40 | value]
    return []

def vex(w, r, addr, m, v, l, p):
    [r, x, b] = ex_transform(r, addr)
    if isinstance(v, GPReg):
        v = v.index
    base = (~v & 15) << 3 | l << 2 | p
    r &= 8
    x &= 8
    b &= 8
    if w or x or b or m != 1:
        return [0xC4, (r ^ 8) << 4 | (x ^ 8) << 3 | (b ^ 8) << 2 | m,
                w << 7 | base]
    return [0xC5, (r ^ 8) << 4 | base]

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
bt_table = {
    'bt':  4,
    'bts': 5,
    'btr': 6,
    'btc': 7,
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
cmov_table = {'cmov' + cond: 0x40 | code for [cond, code] in cond_table.items()}

cond_canon = {cond: conds[0] for conds in all_conds for cond in conds}
canon_table = {prefix + cond: prefix + canon for [cond, canon] in cond_canon.items()
        for prefix in ['j', 'set', 'cmov']}

destructive_ops = {*arg2_table.keys(), *shift_table.keys(), *setcc_table.keys(),
        *cmov_table.keys(), 'imul', 'lea', 'pop'} - {'cmp'}
no_reg_ops = {'cmp', 'test'}

jump_ops = set(list(jump_table.keys()) + ['jmp'])

def normalize_opcode(opcode):
    opcode = opcode.replace('8', '').replace('16', '')
    opcode = opcode.replace('32', '').replace('64', '')

    # Canonicalize instruction names that have multiple names
    if opcode in canon_table:
        opcode = canon_table[opcode]

    return opcode

class Instruction(ASMObj):
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
            sizes = {arg.get_size() for arg in args if isinstance(arg, ASMObj)} - {None}
            if not sizes:
                # XXX default size--this might need more logic later
                size = 64
            else:
                # Mismatched size arguments not supported yet
                assert len(sizes) == 1, sizes
                [size] = sizes

        self.opcode = normalize_opcode(opcode)
        self.size = size
        self.args = [Immediate(arg) if isinstance(arg, int) else arg for arg in args]

    def to_bytes(self):
        w = int(self.size == 64)

        if self.opcode in arg0_table:
            assert not self.args
            return arg0_table[self.opcode]
        elif self.opcode in arg1_table:
            opcode = arg1_table[self.opcode]
            [src] = self.args
            assert isinstance(src, GPReg) or isinstance(src, Address)
            return rex(w, 0, src) + [0xF7] + mod_rm_sib(opcode, src)
        elif self.opcode in arg2_table:
            opcode = arg2_table[self.opcode]
            [dst, src] = self.args

            # Immediates have separate opcodes, so handle them specially here.
            if isinstance(src, Immediate):
                if self.opcode == 'mov':
                    if isinstance(dst, Address):
                        assert fits_32bit(src.value)
                        return rex(w, 0, dst) + [0xC7] + mod_rm_sib(0, dst) + pack32(src)
                    if w:
                        imm_bytes = pack64(src.value)
                    else:
                        imm_bytes = pack32(src.value)
                    return rex(w, 0, dst) + [0xB8 | dst.index & 7] + imm_bytes
                else:
                    assert isinstance(dst, GPReg)
                    [imm_bytes, size_flag] = choose_8_or_32_bit(src.value, 0x2, 0)
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

            assert isinstance(src, GPReg)
            return rex(w, src, dst) + [opcode] + mod_rm_sib(src, dst)
        elif self.opcode in shift_table:
            sub_opcode = shift_table[self.opcode]
            [dst, src] = self.args
            suffix = []
            if isinstance(src, Immediate):
                if src == 1:
                    opcode = 0xD1
                else:
                    opcode = 0xC1
                    suffix = pack8(src.value & 63)
            else:
                assert isinstance(src, GPReg)
                # Only CL
                assert src.index == 1
                opcode = 0xD3
            return rex(w, 0, dst) + [opcode] + mod_rm_sib(sub_opcode, dst) + suffix
        elif self.opcode in bt_table:
            sub_opcode = bt_table[self.opcode]
            [src, bit] = self.args
            assert isinstance(bit, Immediate)
            imm = pack8(bit.value)
            return rex(w, 0, src) + [0x0F, 0xBA] + mod_rm_sib(sub_opcode, src) + imm
        elif self.opcode in cmov_table:
            opcode = cmov_table[self.opcode]
            [dst, src] = self.args
            assert isinstance(dst, GPReg)
            assert isinstance(src, (GPReg, Address))
            return rex(w, dst, src) + [0x0F, opcode] + mod_rm_sib(dst, src)
        elif self.opcode == 'push':
            [src] = self.args
            if isinstance(src, Immediate):
                [imm_bytes, opcode] = choose_8_or_32_bit(src.value, 0x6A, 0x68)
                return [opcode] + imm_bytes
            assert isinstance(src, GPReg)
            return rex(0, 0, src) + [0x50 | (src.index & 7)]
        elif self.opcode == 'pop':
            [dst] = self.args
            assert isinstance(dst, GPReg)
            return rex(0, 0, dst) + [0x58 | (dst.index & 7)]
        elif self.opcode == 'imul':
            # XXX only 2-operand version for now
            [dst, src] = self.args
            assert isinstance(src, GPReg) or isinstance(src, Address)
            assert isinstance(dst, GPReg)
            return rex(w, dst, src) + [0x0F, 0xAF] + mod_rm_sib(dst, src)
        elif self.opcode == 'xchg':
            [dst, src] = self.args
            assert isinstance(src, GPReg)
            assert isinstance(dst, GPReg) or isinstance(dst, Address)
            return rex(w, src, dst) + [0x87] + mod_rm_sib(src, dst)
        elif self.opcode == 'lea':
            [dst, src] = self.args
            assert isinstance(dst, GPReg) and isinstance(src, (Address, Label))
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
            force = isinstance(dst, GPReg) and dst.index & 0xC == 4
            return rex(0, 0, dst, force=force) + [0x0F, opcode] + mod_rm_sib(0, dst)
        assert False, self.opcode

    def __str__(self):
        if self.args:
            args = []
            for arg in self.args:
                if isinstance(arg, GPReg):
                    args = args + [str(arg)]
                elif isinstance(arg, Address):
                    # lea doesn't need a size since it's only the address...
                    use_size_prefix = self.opcode != 'lea'
                    args = args + [arg.to_str(use_size_prefix, self.size)]
                else:
                    assert isinstance(arg, (Immediate, Label)), arg
                    #arg = arg.value
                    if self.opcode in shift_table:
                        arg = arg.value & 63
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
    def create_spec(inst, *args, size=None):
        new_args = []
        for arg in args:
            new_args.append([t if isinstance(t, ASMObj) else (t, size)
                for t in arg])
        inst = '{}{}'.format(inst, size) if size else inst
        return [inst, *new_args]

    for inst in arg0_table.keys():
        yield [inst]

    for size in [32, 64]:
        # Lambda to pass the local 'size' variable
        sspec = lambda *args, size=size: create_spec(*args, size=size)

        for inst in arg1_table.keys():
            yield sspec(inst, 'ra')

        for inst in arg2_table.keys():
            yield sspec(inst, 'r', 'rai')
            yield sspec(inst, 'a', 'r')

        for inst in shift_table.keys():
            yield sspec(inst, 'ra', 'i')
            # Printing shifts by CL is special, since it uses two different
            # register sizes
            yield sspec(inst, 'ra', [GPReg(1, size=8)])

        for inst in bt_table.keys():
            yield sspec(inst, 'ra', 'b')

        for [cond, code] in cmov_table.items():
            yield sspec(cond, 'r', 'ra')

        yield sspec('imul', 'r', 'ra')
        yield sspec('xchg', 'ra', 'r')
        yield sspec('lea', 'r', 'a')
        yield sspec('test', 'ra', 'r')

    # Push/pop don't have a 32-bit operand encoding in 64-bit mode...
    yield create_spec('push', 'ri', size=64)
    yield create_spec('pop', 'r', size=64)

    # Make sure all of the condition codes are tested, to test canonicalization
    for [cond, code] in jump_table.items():
        yield create_spec(cond, 'l', size=None)

    for [cond, code] in setcc_table.items():
        yield create_spec(cond, 'ra', size=8)

    for inst in ['jmp', 'call']:
        yield create_spec(inst, 'r', size=64)
        yield create_spec(inst, 'l', size=None)

    for size in [32, 64]:
        # Lambda to pass the local 'size' variable
        sspec = lambda *args, size=size: create_spec(*args, size=size)

        for inst in bmi_arg2_table.keys():
            yield sspec(inst, 'r', 'ra')

        for inst in bmi_arg3_table.keys():
            [src1, src2] = ['ra', 'r']
            if inst in bmi_arg3_reversed:
                [src2, src1] = [src1, src2]
            yield sspec(inst, 'r', src1, src2)
