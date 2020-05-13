import enum
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

# Dummy abstract base class
class Register(ASMObj): pass

# General-purpose register
class GPReg(Register):
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

# Vector registers
class XMMReg(Register):
    def __init__(self, index: int):
        self.index = index
    def get_size(self):
        return 128
    def __str__(self):
        return 'xmm{}'.format(self.index)
class YMMReg(Register):
    def __init__(self, index: int):
        self.index = index
    def get_size(self):
        return 256
    def __str__(self):
        return 'ymm{}'.format(self.index)
class ZMMReg(Register):
    def __init__(self, index: int):
        self.index = index
    def get_size(self):
        return 512
    def __str__(self):
        return 'zmm{}'.format(self.index)

class Address(ASMObj):
    def __init__(self, base: int, scale: int, index: int, disp: int, size: int=None):
        self.base = base
        self.scale = scale
        self.index = index
        self.disp = disp
        self.size = size

    def get_size(self):
        return self.size

    def to_str(self, use_size_prefix: bool):
        if use_size_prefix:
            size_str = {8: 'BYTE', 16: 'WORD', 32: 'DWORD', 64: 'QWORD'}[self.size]
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
    if isinstance(reg, Register):
        reg = reg.index
    sib_bytes = []
    disp_bytes = []
    if isinstance(rm, Register):
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

    if isinstance(r, Register):
        r = r.index
    if isinstance(x, Register):
        x = x.index
    if isinstance(b, Register):
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
    if isinstance(v, Register):
        v = v.index
    assert isinstance(m, OPF)
    assert isinstance(p, SPF)
    base = (~v & 15) << 3 | l << 2 | p
    r &= 8
    x &= 8
    b &= 8
    if w or x or b or m != 1:
        return [0xC4, (r ^ 8) << 4 | (x ^ 8) << 3 | (b ^ 8) << 2 | m,
                w << 7 | base]
    return [0xC5, (r ^ 8) << 4 | base]

# Basic x86 ops
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

# Opcode extension prefix--these generate extra opcode bytes with SSE, or get packed
# into bitfields of the VEX/EVEX prefixes
OPF = enum.IntEnum('OPF', 'x0F x0Fx38 x0Fx3A', start=1)
OPCODE_PREFIX_BYTES = {OPF.x0F: [0x0F], OPF.x0Fx38: [0x0F, 0x38], OPF.x0Fx3A: [0x0F, 0x3A]}

# Size prefix enum, same deal as opcode prefixes
SPF = enum.IntEnum('SPF', 'none x66 xF3 xF2', start=0)
SIZE_PREFIX_BYTES = {SPF.none: [], SPF.x66: [0x66], SPF.xF3: [0xF3], SPF.xF2: [0xF2]}

# Flags for instruction form
FLAG_3OP = 1
FLAG_IMM = 2

# SSE instructions
sse_table = {
    'addpd':        [1, SPF.x66,  OPF.x0F,    0x58],
    'addps':        [1, SPF.none, OPF.x0F,    0x58],
    'addsubpd':     [1, SPF.x66,  OPF.x0F,    0xD0],
    'addsubps':     [1, SPF.xF2,  OPF.x0F,    0xD0],
    'blendpd':      [3, SPF.x66,  OPF.x0Fx3A, 0x0D],
    'blendps':      [3, SPF.x66,  OPF.x0Fx3A, 0x0C],
    'divpd':        [1, SPF.x66,  OPF.x0F,    0x5E],
    'divps':        [1, SPF.none, OPF.x0F,    0x5E],
    'haddpd':       [1, SPF.x66,  OPF.x0F,    0x7C],
    'haddps':       [1, SPF.xF2,  OPF.x0F,    0x7C],
    'hsubpd':       [1, SPF.x66,  OPF.x0F,    0x7D],
    'hsubps':       [1, SPF.xF2,  OPF.x0F,    0x7D],
    'maxpd':        [1, SPF.x66,  OPF.x0F,    0x5F],
    'maxps':        [1, SPF.none, OPF.x0F,    0x5F],
    'minpd':        [1, SPF.x66,  OPF.x0F,    0x5D],
    'minps':        [1, SPF.none, OPF.x0F,    0x5D],
    'mpsadbw':      [3, SPF.x66,  OPF.x0Fx3A, 0x42],
    'mulpd':        [1, SPF.x66,  OPF.x0F,    0x59],
    'mulps':        [1, SPF.none, OPF.x0F,    0x59],
    'pabsb':        [0, SPF.x66,  OPF.x0Fx38, 0x1C],
    'pabsd':        [0, SPF.x66,  OPF.x0Fx38, 0x1E],
    'pabsw':        [0, SPF.x66,  OPF.x0Fx38, 0x1D],
    'packssdw':     [1, SPF.x66,  OPF.x0F,    0x6B],
    'packsswb':     [1, SPF.x66,  OPF.x0F,    0x63],
    'packusdw':     [1, SPF.x66,  OPF.x0Fx38, 0x2B],
    'packuswb':     [1, SPF.x66,  OPF.x0F,    0x67],
    'paddb':        [1, SPF.x66,  OPF.x0F,    0xFC],
    'paddd':        [1, SPF.x66,  OPF.x0F,    0xFE],
    'paddq':        [1, SPF.x66,  OPF.x0F,    0xD4],
    'paddsb':       [1, SPF.x66,  OPF.x0F,    0xEC],
    'paddsw':       [1, SPF.x66,  OPF.x0F,    0xED],
    'paddusb':      [1, SPF.x66,  OPF.x0F,    0xDC],
    'paddusw':      [1, SPF.x66,  OPF.x0F,    0xDD],
    'paddw':        [1, SPF.x66,  OPF.x0F,    0xFD],
    'palignr':      [3, SPF.x66,  OPF.x0Fx3A, 0x0F],
    'pavgb':        [1, SPF.x66,  OPF.x0F,    0xE0],
    'pavgw':        [1, SPF.x66,  OPF.x0F,    0xE3],
    'pblendw':      [3, SPF.x66,  OPF.x0Fx3A, 0x0E],
    'pcmpeqb':      [1, SPF.x66,  OPF.x0F,    0x74],
    'pcmpeqd':      [1, SPF.x66,  OPF.x0F,    0x76],
    'pcmpeqq':      [1, SPF.x66,  OPF.x0Fx38, 0x29],
    'pcmpeqw':      [1, SPF.x66,  OPF.x0F,    0x75],
    'pcmpgtb':      [1, SPF.x66,  OPF.x0F,    0x64],
    'pcmpgtd':      [1, SPF.x66,  OPF.x0F,    0x66],
    'pcmpgtw':      [1, SPF.x66,  OPF.x0F,    0x65],
    'phaddd':       [1, SPF.x66,  OPF.x0Fx38, 0x02],
    'phaddsw':      [1, SPF.x66,  OPF.x0Fx38, 0x03],
    'phaddw':       [1, SPF.x66,  OPF.x0Fx38, 0x01],
    'phminposuw':   [0, SPF.x66,  OPF.x0Fx38, 0x41],
    'phsubd':       [1, SPF.x66,  OPF.x0Fx38, 0x06],
    'phsubsw':      [1, SPF.x66,  OPF.x0Fx38, 0x07],
    'phsubw':       [1, SPF.x66,  OPF.x0Fx38, 0x05],
    'pmaddubsw':    [1, SPF.x66,  OPF.x0Fx38, 0x04],
    'pmaddwd':      [1, SPF.x66,  OPF.x0F,    0xF5],
    'pmaxsb':       [1, SPF.x66,  OPF.x0Fx38, 0x3C],
    'pmaxsd':       [1, SPF.x66,  OPF.x0Fx38, 0x3D],
    'pmaxsw':       [1, SPF.x66,  OPF.x0F,    0xEE],
    'pmaxub':       [1, SPF.x66,  OPF.x0F,    0xDE],
    'pmaxud':       [1, SPF.x66,  OPF.x0Fx38, 0x3F],
    'pmaxuw':       [1, SPF.x66,  OPF.x0Fx38, 0x3E],
    'pminsb':       [1, SPF.x66,  OPF.x0Fx38, 0x38],
    'pminsd':       [1, SPF.x66,  OPF.x0Fx38, 0x39],
    'pminsw':       [1, SPF.x66,  OPF.x0F,    0xEA],
    'pminub':       [1, SPF.x66,  OPF.x0F,    0xDA],
    'pminud':       [1, SPF.x66,  OPF.x0Fx38, 0x3B],
    'pminuw':       [1, SPF.x66,  OPF.x0Fx38, 0x3A],
    'pmuldq':       [1, SPF.x66,  OPF.x0Fx38, 0x28],
    'pmulhrsw':     [1, SPF.x66,  OPF.x0Fx38, 0x0B],
    'pmulhuw':      [1, SPF.x66,  OPF.x0F,    0xE4],
    'pmulhw':       [1, SPF.x66,  OPF.x0F,    0xE5],
    'pmulld':       [1, SPF.x66,  OPF.x0Fx38, 0x40],
    'pmullw':       [1, SPF.x66,  OPF.x0F,    0xD5],
    'pmuludq':      [1, SPF.x66,  OPF.x0F,    0xF4],
    'psadbw':       [1, SPF.x66,  OPF.x0F,    0xF6],
    'pshufb':       [1, SPF.x66,  OPF.x0Fx38, 0x00],
    'pshufd':       [2, SPF.x66,  OPF.x0F,    0x70],
    'pshufhw':      [2, SPF.xF3,  OPF.x0F,    0x70],
    'pshuflw':      [2, SPF.xF2,  OPF.x0F,    0x70],
    'psignb':       [1, SPF.x66,  OPF.x0Fx38, 0x08],
    'psignd':       [1, SPF.x66,  OPF.x0Fx38, 0x0A],
    'psignw':       [1, SPF.x66,  OPF.x0Fx38, 0x09],
    'psubb':        [1, SPF.x66,  OPF.x0F,    0xF8],
    'psubd':        [1, SPF.x66,  OPF.x0F,    0xFA],
    'psubq':        [1, SPF.x66,  OPF.x0F,    0xFB],
    'psubsb':       [1, SPF.x66,  OPF.x0F,    0xE8],
    'psubsw':       [1, SPF.x66,  OPF.x0F,    0xE9],
    'psubusb':      [1, SPF.x66,  OPF.x0F,    0xD8],
    'psubusw':      [1, SPF.x66,  OPF.x0F,    0xD9],
    'psubw':        [1, SPF.x66,  OPF.x0F,    0xF9],
    'rcpps':        [0, SPF.none, OPF.x0F,    0x53],
    'roundpd':      [2, SPF.x66,  OPF.x0Fx3A, 0x09],
    'roundps':      [2, SPF.x66,  OPF.x0Fx3A, 0x08],
    'rsqrtps':      [0, SPF.none, OPF.x0F,    0x52],
    'shufpd':       [3, SPF.x66,  OPF.x0F,    0xC6],
    'shufps':       [3, SPF.none, OPF.x0F,    0xC6],
    'sqrtpd':       [0, SPF.x66,  OPF.x0F,    0x51],
    'sqrtps':       [0, SPF.none, OPF.x0F,    0x51],
    'subpd':        [1, SPF.x66,  OPF.x0F,    0x5C],
    'subps':        [1, SPF.none, OPF.x0F,    0x5C],
    'unpckhps':     [1, SPF.none, OPF.x0F,    0x15],
}

# SSE sub instructions: these have in immediate and are encoded differently
sse_sub_table = {
    'pslld':   [0x72, 6],
    'pslldq':  [0x73, 7],
    'psllq':   [0x73, 6],
    'psllw':   [0x71, 6],
    'psrad':   [0x72, 4],
    'psraw':   [0x71, 4],
    'psrld':   [0x72, 2],
    'psrldq':  [0x73, 3],
    'psrlq':   [0x73, 2],
    'psrlw':   [0x71, 2],
}

# AVX only instructions (VEX-encoded SSE instructions are added below)
avx_table = {
    'vpblendd':     [3, SPF.x66,  OPF.x0Fx3A, 0x02, ('w', 0)],
    'vpcmpgtq':     [1, SPF.x66,  OPF.x0Fx38, 0x37],
    'vperm2f128':   [3, SPF.x66,  OPF.x0Fx3A, 0x06, ('w', 0)],
    'vperm2i128':   [3, SPF.x66,  OPF.x0Fx3A, 0x46, ('w', 0)],
    'vpermd':       [1, SPF.x66,  OPF.x0Fx38, 0x36, ('w', 0)],
    'vpermpd':      [2, SPF.x66,  OPF.x0Fx3A, 0x01, ('w', 1)],
    'vpermps':      [1, SPF.x66,  OPF.x0Fx38, 0x16, ('w', 0)],
    'vpermq':       [2, SPF.x66,  OPF.x0Fx3A, 0x00, ('w', 1)],
    'vpsllvd':      [1, SPF.x66,  OPF.x0Fx38, 0x47, ('w', 0)],
    'vpsllvq':      [1, SPF.x66,  OPF.x0Fx38, 0x47, ('w', 1)],
    'vpsravd':      [1, SPF.x66,  OPF.x0Fx38, 0x46, ('w', 0)],
    'vpsrlvd':      [1, SPF.x66,  OPF.x0Fx38, 0x45, ('w', 0)],
    'vpsrlvq':      [1, SPF.x66,  OPF.x0Fx38, 0x45, ('w', 1)],
}

# AVX 128-bit only instructions
avx_blacklist = {
    'phminposuw',
}

# Convert flags varargs in SSE/AVX tables to dicts
for table in [sse_table, avx_table]:
    for [inst, [form, spf, opf, opcode, *flags]] in table.items():
        table[inst] = [form, spf, opf, opcode, dict(flags)]

# Add VEX extensions of SSE instructions
for [opcode, value] in sse_table.items():
    if opcode not in avx_blacklist:
        avx_table['v' + opcode] = value

# BMI instructions
bmi_arg2_table = {
    'lzcnt': 0xBD,
    'popcnt': 0xB8,
    'tzcnt': 0xBC,
}
bmi_arg3_table = {
    'andn':  [SPF.none, OPF.x0Fx38, 0xF2],
    'bextr': [SPF.none, OPF.x0Fx38, 0xF7],
    'bzhi':  [SPF.none, OPF.x0Fx38, 0xF5],
    'mulx':  [SPF.xF2,  OPF.x0Fx38, 0xF6],
    'pdep':  [SPF.xF2,  OPF.x0Fx38, 0xF5],
    'pext':  [SPF.xF3,  OPF.x0Fx38, 0xF5],
    'sarx':  [SPF.xF3,  OPF.x0Fx38, 0xF7],
    'shlx':  [SPF.x66,  OPF.x0Fx38, 0xF7],
    'shrx':  [SPF.xF2,  OPF.x0Fx38, 0xF7],
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
    # Canonicalize instruction names that have multiple names
    if opcode in canon_table:
        opcode = canon_table[opcode]

    return opcode

class Instruction(ASMObj):
    def __init__(self, opcode: str, *args):
        self.opcode = normalize_opcode(opcode)
        self.args = [Immediate(arg) if isinstance(arg, int) else arg for arg in args]

    def to_bytes(self):
        # Compute default w value for REX prefix. This isn't always needed, and it's a
        # tad messy, but it's used a bunch of times below, so get it now
        sizes = {arg.get_size() for arg in self.args if isinstance(arg, ASMObj)} - {None}
        size = None
        if not sizes:
            # XXX default size--this might need more logic later
            size = 64
        elif len(sizes) == 1:
            [size] = sizes
        else:
            # Mismatched size arguments--what to do?
            size = self.args[0].get_size()

        w = int(size == 64)

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

        elif self.opcode in sse_table:
            [form, spf, opf, opcode, flags] = sse_table[self.opcode]
            # Check instruction flags for w override
            if 'w' in flags:
                w = flags['w']

            # Convert size/opcode prefix enums to prefix bytes
            spf = SIZE_PREFIX_BYTES[spf]
            opf = OPCODE_PREFIX_BYTES[opf]

            # Parse immediate
            args = self.args.copy()
            imm = []
            if form & FLAG_IMM:
                imm = args.pop()
                assert isinstance(imm, Immediate), imm
                imm = pack8(imm.value)

            assert len(args) == 2, self
            [dst, src] = args
            assert isinstance(src, (XMMReg, Address)), self
            return spf + rex(w, dst, src) + opf + [opcode] + mod_rm_sib(dst, src) + imm
        elif self.opcode in avx_table:
            [form, spf, opf, opcode, flags] = avx_table[self.opcode]
            # Check instruction flags for w override
            if 'w' in flags:
                w = flags['w']

            # Parse immediate
            args = self.args.copy()
            imm = []
            if form & FLAG_IMM:
                imm = args.pop()
                assert isinstance(imm, Immediate), imm
                imm = pack8(imm.value)

            # Encode either 2-op or 3-op instruction
            if form & FLAG_3OP:
                assert len(args) == 3, (self, args)
                [dst, src1, src2] = args
                assert isinstance(src1, (YMMReg, Address)), self
                assert isinstance(src2, (YMMReg, Address)), self
                return vex(w, dst, src2, opf, src1, 1, spf) + [opcode] + mod_rm_sib(dst, src2) + imm
            else:
                assert len(args) == 2, self
                [dst, src] = args
                assert isinstance(src, (YMMReg, Address)), self
                return vex(w, dst, src, opf, 0, 1, spf) + [opcode] + mod_rm_sib(dst, src) + imm
        elif self.opcode in bmi_arg2_table:
            opcode = bmi_arg2_table[self.opcode]
            [dst, src] = self.args
            return [0xF3] + rex(w, dst, src) + [0x0F, opcode] + mod_rm_sib(dst, src)
        elif self.opcode in bmi_arg3_table:
            [spf, opf, opcode] = bmi_arg3_table[self.opcode]
            [dst, src1, src2] = self.args
            if self.opcode in bmi_arg3_reversed:
                [src2, src1] = [src1, src2]
            return vex(w, dst, src1, opf, src2, 0, spf) + [opcode] + mod_rm_sib(dst, src1)
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
                if isinstance(arg, Register):
                    args = args + [str(arg)]
                elif isinstance(arg, Address):
                    # lea doesn't need a size since it's only the address...
                    use_size_prefix = self.opcode != 'lea'
                    args = args + [arg.to_str(use_size_prefix)]
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
    SIZE_TABLE = {
        # GP regs
        'q': 64,
        'd': 32,
        'w': 16,
        'b': 8,
        # V regs
        'x': 128,
        'y': 256,
        'z': 512,
        # Address, label
        'Q': 64,
        'D': 32,
        'W': 16,
        'B': 8,
        'l': None,
        # Immediates
        'i': 8,
        'I': 32,
    }

    inst_list = []
    def add(inst, *args):
        nonlocal inst_list
        new_args = []
        for arg in args:
            new_args.append([t if isinstance(t, ASMObj) else (t, SIZE_TABLE[t])
                for t in arg])
        #inst = '{}{}'.format(inst, size) if size else inst
        inst_list.append((inst, *new_args))

    def add_32_64(inst, *args):
        add(inst, *args)
        add(inst, *(a.replace('q', 'd').replace('Q', 'D')
                if isinstance(a, str) else a for a in args))

    for inst in arg0_table.keys():
        add(inst)

    for inst in arg1_table.keys():
        add_32_64(inst, 'qQ')

    for inst in arg2_table.keys():
        add_32_64(inst, 'q', 'qQi')
        add_32_64(inst, 'Q', 'q')

    for inst in shift_table.keys():
        add_32_64(inst, 'qQ', 'i')
        # Shifts by CL is special, only one register allowed
        add_32_64(inst, 'qQ', [GPReg(1, size=8)])

    for inst in bt_table.keys():
        add_32_64(inst, 'qQ', 'i')

    for [cond, code] in cmov_table.items():
        add_32_64(cond, 'q', 'qQ')

    add_32_64('imul', 'q', 'qQ')
    add_32_64('xchg', 'qQ', 'q')
    add_32_64('lea', 'q', 'Q')
    add_32_64('test', 'qQ', 'q')

    # Push/pop don't have a 32-bit operand encoding in 64-bit mode...
    add('push', 'qi')
    add('pop', 'q')

    # Make sure all of the condition codes are tested, to test canonicalization
    for [cond, code] in jump_table.items():
        add(cond, 'l')

    for [cond, code] in setcc_table.items():
        add(cond, 'bB')

    for inst in ['jmp', 'call']:
        add(inst, 'q')
        add(inst, 'l')

    # SSE

    for [inst, [form, *_]] in sse_table.items():
        args = ['x', 'x']
        if form & FLAG_IMM:
            args.append('i')
        add(inst, *args)

    # AVX

    for [inst, [form, *_]] in avx_table.items():
        args = ['y', 'y']
        if form & FLAG_3OP:
            args.append('y')
        if form & FLAG_IMM:
            args.append('i')
        add(inst, *args)

    for inst in bmi_arg2_table.keys():
        add_32_64(inst, 'q', 'qQ')

    for inst in bmi_arg3_table.keys():
        [src1, src2] = ['qQ', 'q']
        if inst in bmi_arg3_reversed:
            [src2, src1] = [src1, src2]
        add_32_64(inst, 'q', src1, src2)

    return inst_list
