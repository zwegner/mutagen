import enum
import itertools

################################################################################
## Assembler types #############################################################
################################################################################

# Dummy abstract base class
class ASMObj: pass

# Base class for all address forms, both normal Addresses as well as Label/Data
class BaseAddress(ASMObj): pass

class Label(BaseAddress):
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
    def __init__(self, label, size: int, offset: int=0):
        self.label = label
        self.size = size
        self.offset = offset

class Immediate(ASMObj):
    def __init__(self, value: int, size: int=None):
        self.value = value
        self.size = size
    def get_size(self):
        return self.size
    def __str__(self):
        return str(self.value)

class Data(BaseAddress):
    def __init__(self, data_bytes: list, alignment: int=1):
        self.data_bytes = data_bytes
        assert not alignment & alignment - 1
        self.alignment = alignment
    def __str__(self):
        return '<data %s>' % self.data_bytes
    def get_size(self):
        return None

# Base class for registers
class Register(ASMObj):
    def _key(self):
        return (type(self), self.index, self.size)
    def __hash__(self):
        return hash(self._key())
    def __eq__(self, other):
        return self._key() == other._key()
    def __lt__(self, other):
        assert type(self) is type(other)
        return self.index < other.index

# General-purpose register
class GPReg(Register):
    def __init__(self, index: int, size: int=None):
        self.index = index
        self.size = size or 64 # XXX default of 64

    def get_size(self):
        return self.size

    def __repr__(self):
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
class VecReg(Register):
    def __init__(self, index: int, size: int=None):
        self.index = index
        self.size = size or VEC_SIZE_BITS
    def get_size(self):
        return self.size
    def __repr__(self):
        return '{}mm{}'.format(VECTOR_PREFIX[self.size], self.index)

class Address(BaseAddress):
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
            size_str = ADDR_SIZE_TABLE[self.size]
            size_str = size_str + ' PTR '
        else:
            size_str = ''
        # XXX this SIB stuff assumes GPR, doesn't handle gather/scatter
        base = self.base
        if isinstance(base, int):
            base = GPReg(self.base, size=64)
        parts = [base]
        if self.scale:
            parts = parts + ['{}*{}'.format(GPReg(self.index, size=64),
                self.scale)]
        if self.disp:
            parts = parts + [self.disp]
        return '{}[{}]'.format(size_str, '+'.join(map(str, parts)))
    def __repr__(self):
        return 'Address({}, {}, {}, {})'.format(self.base, self.scale, self.index, self.disp)

class Instruction(ASMObj):
    def __init__(self, mnem: str, *args):
        self.mnem = canon_table.get(mnem, mnem)
        self.args = [Immediate(arg) if isinstance(arg, int) else arg for arg in args]

    def __str__(self):
        if self.args:
            args = []
            for arg in self.args:
                if isinstance(arg, Register):
                    args = args + [str(arg)]
                elif isinstance(arg, Address):
                    # lea doesn't need a size since it's only the address...
                    use_size_prefix = self.mnem != 'lea'
                    args = args + [arg.to_str(use_size_prefix)]
                else:
                    assert isinstance(arg, (Immediate, Label, Data)), arg
                    if self.mnem in shift_table:
                        arg = arg.value & 63
                    # XXX handle different immediate sizes (to convert to unsigned)
                    args = args + [str(arg)]
            argstr = ' ' + ','.join(args)
        else:
            argstr = ''
        return self.mnem + argstr

################################################################################
## Assembler constants #########################################################
################################################################################

# Vector register size used by default (for now, we're targeting AVX2).
# Eventually this should be configurable/dynamic based on target
VEC_SIZE_BYTES = 32
VEC_SIZE_BITS = VEC_SIZE_BYTES * 8
# ...and vector move instruction to match.
VEC_MOVE = lambda dst, src: Instruction('vmovdqu', dst, src)

ADDR_SIZE_TABLE = {
    8: 'BYTE',
    16: 'WORD',
    32: 'DWORD',
    64: 'QWORD',
    128: 'XMMWORD',
    256: 'YMMWORD',
}

VECTOR_PREFIX = {
    128: 'x',
    256: 'y',
    512: 'z'
}

# One-char shortcut table for specifying operand type/size
ARG_TYPE_TABLE = {
    # GP regs
    'b': (GPReg, 8),
    'w': (GPReg, 16),
    'd': (GPReg, 32),
    'q': (GPReg, 64),
    # V regs
    'x': (VecReg, 128),
    'y': (VecReg, 256),
    'z': (VecReg, 512),
    # Address, label
    'B': (BaseAddress, 8),
    'W': (BaseAddress, 16),
    'D': (BaseAddress, 32),
    'Q': (BaseAddress, 64),
    'X': (BaseAddress, 128),
    'Y': (BaseAddress, 256),
    'Z': (BaseAddress, 512),
    'l': (Label, None),
    # Immediates
    'i': (Immediate, 8),
    'I': (Immediate, 32),
}

# Conditions
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

# Opcode extension prefix--these generate extra opcode bytes with SSE, or get
# packed into bitfields of the VEX/EVEX prefixes
OPF = enum.IntEnum('OPF', 'x0F x0Fx38 x0Fx3A', start=1)
OPCODE_PREFIX_BYTES = {OPF.x0F: [0x0F], OPF.x0Fx38: [0x0F, 0x38],
        OPF.x0Fx3A: [0x0F, 0x3A]}

# Size prefix enum, same deal as opcode prefixes
SPF = enum.IntEnum('SPF', 'none x66 xF3 xF2', start=0)
SIZE_PREFIX_BYTES = {SPF.none: [], SPF.x66: [0x66], SPF.xF3: [0xF3],
        SPF.xF2: [0xF2]}

# Flags for SSE/AVX instruction form
FLAG_3OP = 1
FLAG_IMM = 2

################################################################################
## Instruction tables ##########################################################
################################################################################

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
bs_table = {
    'bsf': 0xBC,
    'bsr': 0xBD,
}

# Condition-using instructions
jump_table = {'j' + cond: 0x80 | code for [cond, code] in cond_table.items()}
setcc_table = {'set' + cond: 0x90 | code for [cond, code] in cond_table.items()}
cmov_table = {'cmov' + cond: 0x40 | code for [cond, code] in cond_table.items()}

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
    'dppd':         [3, SPF.x66,  OPF.x0Fx3A, 0x41, ('vlen', 128)],
    'dpps':         [3, SPF.x66,  OPF.x0Fx3A, 0x40],
    'extractps':    [2, SPF.x66,  OPF.x0Fx3A, 0x17, ('vlen', 128),
            ('reverse', True), ('arg_types', ('d', None)), ('addr_size', 32)],
    'haddpd':       [1, SPF.x66,  OPF.x0F,    0x7C],
    'haddps':       [1, SPF.xF2,  OPF.x0F,    0x7C],
    'hsubpd':       [1, SPF.x66,  OPF.x0F,    0x7D],
    'hsubps':       [1, SPF.xF2,  OPF.x0F,    0x7D],
    'insertps':     [3, SPF.x66,  OPF.x0Fx3A, 0x21, ('vlen', 128),
            ('addr_size', 32)],
    'maxpd':        [1, SPF.x66,  OPF.x0F,    0x5F],
    'maxps':        [1, SPF.none, OPF.x0F,    0x5F],
    'minpd':        [1, SPF.x66,  OPF.x0F,    0x5D],
    'minps':        [1, SPF.none, OPF.x0F,    0x5D],
    'movdqa':       [0, SPF.x66,  OPF.x0F,    0x6F, ('reverse_opcode', 0x7F)],
    'movdqu':       [0, SPF.xF3,  OPF.x0F,    0x6F, ('reverse_opcode', 0x7F)],
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
    'pand':         [1, SPF.x66,  OPF.x0F,    0xDB],
    'pandn':        [1, SPF.x66,  OPF.x0F,    0xDF],
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
    'pextrb':       [2, SPF.x66,  OPF.x0Fx3A, 0x14, ('vlen', 128),
            ('reverse', True), ('arg_types', ('d', None)), ('addr_size', 8)],
    'pextrd':       [2, SPF.x66,  OPF.x0Fx3A, 0x16, ('w', 0), ('vlen', 128),
            ('reverse', True), ('arg_types', ('d', None)), ('addr_size', 32)],
    'pextrq':       [2, SPF.x66,  OPF.x0Fx3A, 0x16, ('w', 1), ('vlen', 128),
            ('reverse', True), ('arg_types', ('q', None)), ('addr_size', 64)],
    'pextrw':       [2, SPF.x66,  OPF.x0Fx3A, 0x15, ('vlen', 128),
            ('reverse', True), ('arg_types', ('d', None)), ('addr_size', 16)],
    'phaddd':       [1, SPF.x66,  OPF.x0Fx38, 0x02],
    'phaddsw':      [1, SPF.x66,  OPF.x0Fx38, 0x03],
    'phaddw':       [1, SPF.x66,  OPF.x0Fx38, 0x01],
    'phminposuw':   [0, SPF.x66,  OPF.x0Fx38, 0x41, ('vlen', 128)],
    'phsubd':       [1, SPF.x66,  OPF.x0Fx38, 0x06],
    'phsubsw':      [1, SPF.x66,  OPF.x0Fx38, 0x07],
    'phsubw':       [1, SPF.x66,  OPF.x0Fx38, 0x05],
    'pinsrb':       [3, SPF.x66,  OPF.x0Fx3A, 0x20, ('vlen', 128),
            ('arg_types', (None, None, 'd')), ('addr_size', 8)],
    'pinsrd':       [3, SPF.x66,  OPF.x0Fx3A, 0x22, ('w', 0), ('vlen', 128),
            ('arg_types', (None, None, 'd')), ('addr_size', 32)],
    'pinsrq':       [3, SPF.x66,  OPF.x0Fx3A, 0x22, ('w', 1), ('vlen', 128),
            ('arg_types', (None, None, 'q')), ('addr_size', 64)],
    'pinsrw':       [3, SPF.x66,  OPF.x0F,    0xC4, ('vlen', 128),
            ('arg_types', (None, None, 'd')), ('addr_size', 16)],
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
    'pmovmskb':     [0, SPF.x66,  OPF.x0F,    0xD7,
            ('arg_types', ('d', None)), ('reg_only', True)],
    'pmuldq':       [1, SPF.x66,  OPF.x0Fx38, 0x28],
    'pmulhrsw':     [1, SPF.x66,  OPF.x0Fx38, 0x0B],
    'pmulhuw':      [1, SPF.x66,  OPF.x0F,    0xE4],
    'pmulhw':       [1, SPF.x66,  OPF.x0F,    0xE5],
    'pmulld':       [1, SPF.x66,  OPF.x0Fx38, 0x40],
    'pmullw':       [1, SPF.x66,  OPF.x0F,    0xD5],
    'pmuludq':      [1, SPF.x66,  OPF.x0F,    0xF4],
    'por':          [1, SPF.x66,  OPF.x0F,    0xEB],
    'psadbw':       [1, SPF.x66,  OPF.x0F,    0xF6],
    'pshufb':       [1, SPF.x66,  OPF.x0Fx38, 0x00],
    'pshufd':       [2, SPF.x66,  OPF.x0F,    0x70],
    'pshufhw':      [2, SPF.xF3,  OPF.x0F,    0x70],
    'pshuflw':      [2, SPF.xF2,  OPF.x0F,    0x70],
    'psignb':       [1, SPF.x66,  OPF.x0Fx38, 0x08],
    'psignd':       [1, SPF.x66,  OPF.x0Fx38, 0x0A],
    'psignw':       [1, SPF.x66,  OPF.x0Fx38, 0x09],
    'pslld':        [1, SPF.x66,  OPF.x0F,    0xF2, ('sub_opcode', (0x72, 6)),
            ('arg_types', (None, None, 'x')), ('addr_size', 128)],
    'pslldq':       [1, SPF.x66,  OPF.x0F,    0x73, ('sub_opcode', (0x73, 7)),
            ('imm_only', True)],
    'psllq':        [1, SPF.x66,  OPF.x0F,    0xF3, ('sub_opcode', (0x73, 6)),
            ('arg_types', (None, None, 'x')), ('addr_size', 128)],
    'psllw':        [1, SPF.x66,  OPF.x0F,    0xF1, ('sub_opcode', (0x71, 6)),
            ('arg_types', (None, None, 'x')), ('addr_size', 128)],
    'psrad':        [1, SPF.x66,  OPF.x0F,    0xE2, ('sub_opcode', (0x72, 4)),
            ('arg_types', (None, None, 'x')), ('addr_size', 128)],
    'psraw':        [1, SPF.x66,  OPF.x0F,    0xE1, ('sub_opcode', (0x71, 4)),
            ('arg_types', (None, None, 'x')), ('addr_size', 128)],
    'psrld':        [1, SPF.x66,  OPF.x0F,    0xD2, ('sub_opcode', (0x72, 2)),
            ('arg_types', (None, None, 'x')), ('addr_size', 128)],
    'psrldq':       [1, SPF.x66,  OPF.x0F,    0x73, ('sub_opcode', (0x73, 3)),
            ('imm_only', True)],
    'psrlq':        [1, SPF.x66,  OPF.x0F,    0xD3, ('sub_opcode', (0x73, 2)),
            ('arg_types', (None, None, 'x')), ('addr_size', 128)],
    'psrlw':        [1, SPF.x66,  OPF.x0F,    0xD1, ('sub_opcode', (0x71, 2)),
            ('arg_types', (None, None, 'x')), ('addr_size', 128)],
    'psubb':        [1, SPF.x66,  OPF.x0F,    0xF8],
    'psubd':        [1, SPF.x66,  OPF.x0F,    0xFA],
    'psubq':        [1, SPF.x66,  OPF.x0F,    0xFB],
    'psubsb':       [1, SPF.x66,  OPF.x0F,    0xE8],
    'psubsw':       [1, SPF.x66,  OPF.x0F,    0xE9],
    'psubusb':      [1, SPF.x66,  OPF.x0F,    0xD8],
    'psubusw':      [1, SPF.x66,  OPF.x0F,    0xD9],
    'psubw':        [1, SPF.x66,  OPF.x0F,    0xF9],
    'punpckhbw':    [1, SPF.x66,  OPF.x0F,    0x68],
    'punpckhdq':    [1, SPF.x66,  OPF.x0F,    0x6A],
    'punpckhqdq':   [1, SPF.x66,  OPF.x0F,    0x6D],
    'punpckhwd':    [1, SPF.x66,  OPF.x0F,    0x69],
    'punpcklbw':    [1, SPF.x66,  OPF.x0F,    0x60],
    'punpckldq':    [1, SPF.x66,  OPF.x0F,    0x62],
    'punpcklqdq':   [1, SPF.x66,  OPF.x0F,    0x6C],
    'punpcklwd':    [1, SPF.x66,  OPF.x0F,    0x61],
    'pxor':         [1, SPF.x66,  OPF.x0F,    0xEF],
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
    'unpckhpd':     [1, SPF.x66,  OPF.x0F,    0x15],
    'unpckhps':     [1, SPF.none, OPF.x0F,    0x15],
    'unpcklpd':     [1, SPF.x66,  OPF.x0F,    0x14],
    'unpcklps':     [1, SPF.none, OPF.x0F,    0x14],
}

# AVX only instructions (VEX-encoded SSE instructions are added below)
avx_table = {
    'vbroadcastsd': [0, SPF.x66,  OPF.x0Fx38, 0x19, ('w', 0),
            ('arg_types', (None, 'x')), ('addr_size', 64)],
    'vbroadcastss': [0, SPF.x66,  OPF.x0Fx38, 0x18, ('w', 0),
            ('arg_types', (None, 'x')), ('addr_size', 32)],
    'vextracti128': [2, SPF.x66,  OPF.x0Fx3A, 0x39, ('w', 0),
            ('reverse', True), ('arg_types', ('x', None)), ('addr_size', 128)],
    'vinsertf128':  [3, SPF.x66,  OPF.x0Fx3A, 0x18, ('w', 0),
            ('arg_types', (None, None, 'x')), ('addr_size', 128)],
    'vinserti128':  [3, SPF.x66,  OPF.x0Fx3A, 0x38, ('w', 0),
            ('arg_types', (None, None, 'x')), ('addr_size', 128)],
    'vpblendd':     [3, SPF.x66,  OPF.x0Fx3A, 0x02, ('w', 0)],
    'vpbroadcastb': [0, SPF.x66,  OPF.x0Fx38, 0x78, ('w', 0),
            ('arg_types', (None, 'x')), ('addr_size', 8)],
    'vpbroadcastd': [0, SPF.x66,  OPF.x0Fx38, 0x58, ('w', 0),
            ('arg_types', (None, 'x')), ('addr_size', 32)],
    'vpbroadcastq': [0, SPF.x66,  OPF.x0Fx38, 0x59, ('w', 0),
            ('arg_types', (None, 'x')), ('addr_size', 64)],
    'vpbroadcastw': [0, SPF.x66,  OPF.x0Fx38, 0x79, ('w', 0),
            ('arg_types', (None, 'x')), ('addr_size', 16)],
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

# BMI instructions
bmi_arg2_table = {
    'lzcnt':   0xBD,
    'popcnt':  0xB8,
    'tzcnt':   0xBC,
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
bmi_arg3_reversed = {'andn', 'mulx', 'pdep', 'pext'}

################################################################################
## Spec creation (cleaning up the tables etc.) #################################
################################################################################

# Spec class for storing metadata about instruction forms
class InstSpec:
    def __init__(self, inst, forms, is_destructive=True, is_jump=False,
            needs_register=True, vec_form=None, spf=None, opf=None,
            opcode=None, flags=None):
        self.inst = inst
        self.forms = forms
        self.is_destructive = is_destructive
        self.is_jump = is_jump
        self.needs_register = needs_register
        self.vec_form = vec_form
        self.spf = spf
        self.opf = opf
        self.opcode = opcode
        self.flags = flags

# Create a big table of instruction specs
INST_SPECS = {}

def _add(inst, *args, sizes=[None], change_imm_size=False, **kwargs):
    global INST_SPECS
    if inst not in INST_SPECS:
        INST_SPECS[inst] = InstSpec(inst, [], **kwargs)
    spec = INST_SPECS[inst]
    for form in itertools.product(*args):
        for size in sizes:
            form_args = []
            for t in form:
                if isinstance(t, str):
                    t = ARG_TYPE_TABLE[t]
                    if size is not None and (change_imm_size or t[0] is not Immediate):
                        if size == 64 and t[0] is Immediate and inst != 'mov':
                            t = (t[0], 32)
                        else:
                            t = (t[0], size)
                form_args.append(t)
            spec.forms.append(form_args)

def _add_32_64(inst, *args, **kwargs):
    _add(inst, *args, sizes=[64, 32], **kwargs)

for inst in arg0_table.keys():
    _add(inst, is_destructive=False, needs_register=False)

for inst in arg1_table.keys():
    _add_32_64(inst, 'qQ')

for inst in arg2_table.keys():
    is_write = (inst not in {'cmp', 'test'})
    destr = is_write and inst != 'mov'
    sizes = [64, 32, 16, 8]
    _add(inst, 'q', 'qQ', sizes=sizes, needs_register=is_write, is_destructive=destr)
    _add(inst, 'Q', 'q', sizes=sizes, needs_register=is_write, is_destructive=destr)
    _add(inst, 'q', 'i', sizes=sizes, change_imm_size=True, needs_register=is_write,
            is_destructive=destr)

for inst in shift_table.keys():
    _add_32_64(inst, 'qQ', 'i')
    # Shifts by CL is special, only one register allowed
    _add_32_64(inst, 'qQ', [GPReg(1, size=8)])

for inst in bt_table.keys():
    _add_32_64(inst, 'qQ', 'i')

for inst in bs_table.keys():
    _add_32_64(inst, 'q', 'qQ')

for [cond, code] in cmov_table.items():
    # cmov is considered a destructive op (I guess) as opposed to mov, since
    # it can read the destination
    _add_32_64(cond, 'q', 'qQ')

_add_32_64('imul', 'q', 'qQ')
_add_32_64('xchg', 'qQ', 'q')
_add_32_64('lea', 'q', 'Q', is_destructive=False)
_add_32_64('test', 'qQ', 'q', is_destructive=False, needs_register=False)

# Push/pop don't have a 32-bit operand encoding in 64-bit mode...
_add('push', 'qi', is_destructive=False, needs_register=False)
_add('pop', 'q', is_destructive=False, needs_register=False)

# Make sure all of the condition codes are tested, to test canonicalization
for [cond, code] in jump_table.items():
    _add(cond, 'l', is_jump=True, is_destructive=False, needs_register=False)
_add('jmp', 'ql', is_jump=True, is_destructive=False, needs_register=False)

for [cond, code] in setcc_table.items():
    _add(cond, 'bB', is_destructive=False)

_add('call', 'ql', is_destructive=False, needs_register=False)

# BMI

for inst in bmi_arg2_table.keys():
    _add_32_64(inst, 'q', 'qQ', is_destructive=False)

for inst in bmi_arg3_table.keys():
    [src1, src2] = ['qQ', 'q']
    if inst in bmi_arg3_reversed:
        [src2, src1] = [src1, src2]
    _add_32_64(inst, 'q', src1, src2, is_destructive=False)

# Add VEX extensions of SSE instructions
for [mnem, value] in sse_table.items():
    avx_table['v' + mnem] = value

# Process various flags in the above SSE/AVX tables, to get valid instruction forms
for table in [sse_table, avx_table]:
    for [inst, [vec_form, spf, opf, opcode, *flag_args]] in table.items():
        flags = dict(flag_args)

        # Get extension-specific values
        is_sse = (table is sse_table)
        vlen = 128 if is_sse else 256
        if 'vlen' in flags:
            vlen = flags['vlen']
        vtype = 'x' if vlen == 128 else 'y'

        # Create basic form. Most instructions can have the types inferred from
        # the vec_form (2/3 op, immediate)
        if 'arg_types' in flags:
            types = list(flags.pop('arg_types'))
            if is_sse and vec_form & FLAG_3OP:
                assert types[0] == types[1]
                types = types[1:]
            arg_types = [t or vtype for t in types]
        else:
            arg_types = [vtype, vtype]
            if not is_sse and vec_form & FLAG_3OP:
                arg_types.append(vtype)
        if vec_form & FLAG_IMM:
            arg_types.append('i')

        basic_form = [ARG_TYPE_TABLE[t] for t in arg_types]

        # Create alternate forms (reg, reg; reg, mem; etc.)
        forms = []

        # Only add the basic form if not an immediate-only instruction
        if not flags.get('imm_only'):
            forms.append(basic_form)

        # Create an immediate form if there's a sub-opcode
        if 'sub_opcode' in flags:
            assert not vec_form & FLAG_IMM
            imm_form = basic_form.copy()
            imm_form[-1] = (Immediate, 8)
            forms.append(imm_form)

        # Add address form
        if not flags.get('reg_only') and not flags.get('imm_only'):
            if flags.get('reverse'):
                addr_arg = 0
            elif vec_form & FLAG_IMM:
                addr_arg = len(basic_form) - 2
            else:
                addr_arg = len(basic_form) - 1

            size = flags.get('addr_size', basic_form[addr_arg][1])
            addr_form = basic_form.copy()
            addr_form[addr_arg] = (BaseAddress, size)
            forms.append(addr_form)

            # If there's a reverse_opcode flag, that means the instructions has both
            # reg, mem and mem, reg forms. Add the mem, reg possibility
            if flags.get('reverse_opcode'):
                assert addr_arg != 0
                rev_addr_form = basic_form.copy()
                rev_addr_form[0] = (BaseAddress, size)
                forms.append(rev_addr_form)

        assert inst not in INST_SPECS
        is_destructive = bool(vec_form & FLAG_3OP)
        INST_SPECS[inst] = InstSpec(inst, tuple(forms),
                is_destructive=is_destructive, vec_form=vec_form,
                spf=spf, opf=opf, opcode=opcode, flags=flags)

# Set up a table to canonicalize instructions that have multiple equivalent
# mnemonics (right now, just anything with a condition code)
cond_canon = {cond: conds[0] for conds in all_conds for cond in conds}
canon_table = {prefix + cond: prefix + canon for [cond, canon] in cond_canon.items()
        for prefix in ['j', 'set', 'cmov']}

# Set up instruction argument types, now that all forms have been added
for spec in INST_SPECS.values():
    # Keep tuples of allowable types for each argument
    spec.arg_types = []
    assert len(set(len(f) for f in spec.forms)) == 1
    for args in zip(*spec.forms):
        types = set()
        for arg in args:
            if isinstance(arg, ASMObj):
                arg = type(arg)
            else:
                [arg, _] = arg
            types.add(arg)
        # Add regular int objects if we accept immediates
        if Immediate in types:
            types.add(int)
        spec.arg_types.append(tuple(types))

################################################################################
## Utility functions ###########################################################
################################################################################

def fits_bits(imm: int, size: int):
    limit = 1 << size - 1
    return -limit <= imm <= limit - 1

def pack_bits(imm: int, size: int):
    assert not size & 7
    assert fits_bits(imm, size)
    return [(imm >> shift) & 0xFF for shift in range(0, size, 8)]

def pack8(imm: int):
    # Kinda gross: allow signed or unsigned bytes
    assert -128 <= imm < 256
    return [imm & 0xFF]

# Basic helper function to factor out a common pattern of choosing a 1/4 byte encoding,
# along with another value that depends on which is chosen
def choose_8_or_32_bit(imm, op8, op32):
    if fits_bits(imm, 8):
        return [pack8(imm), op8]
    return [pack_bits(imm, 32), op32]

def mod_rm_sib(reg, rm):
    if isinstance(reg, Register):
        reg = reg.index
    sib_bytes = []
    disp_bytes = []
    if isinstance(rm, Register):
        mod = 3
        base = rm.index
    # Handle relocations with RIP-relative addressing. This assumes anything not
    # a label or data is an address
    elif isinstance(rm, (Label, Data)) or isinstance(rm.base, Data):
        offset = 0
        if isinstance(rm, Address):
            assert not rm.scale
            offset = rm.disp
            rm = rm.base
        mod = 0
        base = 5
        disp_bytes = [Relocation(rm, 4, offset=offset)]
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
            disp_bytes = pack_bits(addr.disp, 32)
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
        if isinstance(b, Data):
            assert not x, 'cannot use index on RIP-relative address'
            [x, b] = [0, 5]
    elif isinstance(addr, (Label, Data)):
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

################################################################################
## The main x86 assembler functionality ########################################
################################################################################

def assemble_inst(inst):
    # Compute default w value for REX prefix. This isn't always needed, and
    # it's a tad messy, but it's used a bunch of times below, so get it now
    sizes = {arg.get_size() for arg in inst.args
            if isinstance(arg, ASMObj)} - {None}
    size = None
    if not sizes:
        # XXX default size--this might need more logic later
        size = 64
    elif len(sizes) == 1:
        [size] = sizes
    else:
        # Mismatched size arguments--what to do?
        size = inst.args[0].get_size()

    # Get spec and check types
    spec = INST_SPECS[inst.mnem]
    for form in spec.forms:
        assert len(form) == len(inst.args), ('wrong number of arguments to %s, '
                'got %s, expected %s' % (inst.mnem, len(inst.args), len(form)))
        if all(isinstance(a, t[0]) and a.get_size() in {None, t[1]}
                if isinstance(t, tuple) else a == t
                for [a, t] in zip(inst.args, form)):
            break
    else:
        assert False, ('argument types to %s %s do not match any of the valid forms '
                'of %s: %s' % (inst, inst.args, inst.mnem, spec.forms))

    w = int(size == 64)

    if inst.mnem in arg0_table:
        return arg0_table[inst.mnem]
    elif inst.mnem in arg1_table:
        opcode = arg1_table[inst.mnem]
        [src] = inst.args
        return rex(w, 0, src) + [0xF7] + mod_rm_sib(opcode, src)
    elif inst.mnem in arg2_table:
        opcode = arg2_table[inst.mnem]
        [dst, src] = inst.args

        use_mod_rm = True
        imm_bytes = []
        prefix = []
        force_rex = 0

        # Handle immediates
        if isinstance(src, Immediate):
            # mov is special
            if inst.mnem == 'mov':
                sub_opcode = 0
                if isinstance(dst, Address):
                    opcode = 0xC6
                else:
                    use_mod_rm = False
                    opcode = 0xB0 | dst.index & 7
                imm_bytes = pack_bits(src.value, src.size or size)
            else:
                sub_opcode = opcode
                opcode = 0x80

                # Special 8-bit sign extension form
                if src.size != 8 and fits_bits(src.value, 8):
                    opcode |= 0x2
                    imm_bytes = pack_bits(src.value, 8)
                else:
                    imm_bytes = pack_bits(src.value, src.size or 32)

            src = 0
        # Non-immediates
        else:
            # Mov is different here too, but can mostly be handled like other ops
            if inst.mnem == 'mov':
                opcode = 0x88
            else:
                opcode <<= 3

            # op reg, mem is handled by flipping the direction bit and
            # swapping src/dst.
            if isinstance(src, BaseAddress):
                opcode |= 0x2
                [src, dst] = [dst, src]

            sub_opcode = src

        # Handle 8/16/32/64-bit operand sizes
        if dst.size == 8:
            force_rex = 1
        elif dst.size in {16, 32, 64}:
            opcode |= 0x1 if use_mod_rm else 0x8
            if dst.size == 16:
                prefix = [0x66]
        else:
            assert False, 'bad size: %s' % dst.size

        mod_rm = mod_rm_sib(sub_opcode, dst) if use_mod_rm else []

        return (prefix + rex(w, src, dst, force=force_rex) + [opcode] +
                mod_rm + imm_bytes)

    elif inst.mnem in shift_table:
        sub_opcode = shift_table[inst.mnem]
        [dst, src] = inst.args
        suffix = []
        if isinstance(src, Immediate):
            if src == 1:
                opcode = 0xD1
            else:
                opcode = 0xC1
                suffix = pack8(src.value & 63)
        else:
            # Only CL
            assert src.index == 1
            opcode = 0xD3
        return rex(w, 0, dst) + [opcode] + mod_rm_sib(sub_opcode, dst) + suffix
    elif inst.mnem in bt_table:
        sub_opcode = bt_table[inst.mnem]
        [src, bit] = inst.args
        imm = pack8(bit.value)
        return rex(w, 0, src) + [0x0F, 0xBA] + mod_rm_sib(sub_opcode, src) + imm
    elif inst.mnem in bs_table:
        opcode = bs_table[inst.mnem]
        [dst, src] = inst.args
        return rex(w, dst, src) + [0x0F, opcode] + mod_rm_sib(dst, src)
    elif inst.mnem in cmov_table:
        opcode = cmov_table[inst.mnem]
        [dst, src] = inst.args
        return rex(w, dst, src) + [0x0F, opcode] + mod_rm_sib(dst, src)
    elif inst.mnem == 'push':
        [src] = inst.args
        if isinstance(src, Immediate):
            [imm_bytes, opcode] = choose_8_or_32_bit(src.value, 0x6A, 0x68)
            return [opcode] + imm_bytes
        return rex(0, 0, src) + [0x50 | (src.index & 7)]
    elif inst.mnem == 'pop':
        [dst] = inst.args
        return rex(0, 0, dst) + [0x58 | (dst.index & 7)]
    elif inst.mnem == 'imul':
        # XXX only 2-operand version for now
        [dst, src] = inst.args
        return rex(w, dst, src) + [0x0F, 0xAF] + mod_rm_sib(dst, src)
    elif inst.mnem == 'xchg':
        [dst, src] = inst.args
        return rex(w, src, dst) + [0x87] + mod_rm_sib(src, dst)
    elif inst.mnem == 'lea':
        [dst, src] = inst.args
        return rex(w, dst, src) + [0x8D] + mod_rm_sib(dst, src)
    elif inst.mnem == 'test':
        # Test has backwards arguments, weird
        [src2, src1] = inst.args
        return rex(w, src1, src2) + [0x85] + mod_rm_sib(src1, src2)

    elif inst.mnem in sse_table:
        # Check instruction flags for w override
        if 'w' in spec.flags:
            w = spec.flags['w']

        opcode = spec.opcode

        # Convert size/opcode prefix enums to prefix bytes
        spf = SIZE_PREFIX_BYTES[spec.spf]
        opf = OPCODE_PREFIX_BYTES[spec.opf]

        # Parse immediate
        if isinstance(inst.args[-1], Immediate):
            if 'sub_opcode' in spec.flags:
                [src, imm] = inst.args
                [opcode, dst] = spec.flags['sub_opcode']
            else:
                assert spec.vec_form & FLAG_IMM
                [dst, src, imm] = inst.args
            imm = pack8(imm.value)
        else:
            assert len(inst.args) == 2, inst
            [dst, src] = inst.args
            imm = []

        if spec.flags.get('reverse'):
            [dst, src] = [src, dst]
        elif 'reverse_opcode' in spec.flags:
            if isinstance(dst, Address):
                [dst, src] = [src, dst]
                opcode = spec.flags['reverse_opcode']

        return (spf + rex(w, dst, src) + opf + [opcode] +
                mod_rm_sib(dst, src) + imm)
    elif inst.mnem in avx_table:
        # Check instruction flags for w override
        w = spec.flags.get('w', w)
        vlen = 0 if spec.flags.get('vlen') == 128 else 1
        opcode = spec.opcode

        # Parse immediate
        args = inst.args.copy()
        imm = []
        if spec.vec_form & FLAG_IMM:
            imm = args.pop()
            imm = pack8(imm.value)

        # Encode either 2-op or 3-op instruction
        if spec.vec_form & FLAG_3OP:
            [dst, src1, src2] = args
            # Encode immediate
            if isinstance(src2, Immediate):
                assert 'sub_opcode' in spec.flags
                assert not imm
                imm = pack8(src2.value)
                [src1, src2] = [dst, src1]
                [opcode, dst] = spec.flags['sub_opcode']
        else:
            [dst, src] = args
            if spec.flags.get('reverse'):
                [dst, src] = [src, dst]
            elif 'reverse_opcode' in spec.flags:
                if isinstance(dst, Address):
                    [dst, src] = [src, dst]
                    opcode = spec.flags['reverse_opcode']
            # x86 encodings are weird
            [src1, src2] = [0, src]

        return (vex(w, dst, src2, spec.opf, src1, vlen, spec.spf) + [opcode] +
                mod_rm_sib(dst, src2) + imm)
    elif inst.mnem in bmi_arg2_table:
        opcode = bmi_arg2_table[inst.mnem]
        [dst, src] = inst.args
        return [0xF3] + rex(w, dst, src) + [0x0F, opcode] + mod_rm_sib(dst, src)
    elif inst.mnem in bmi_arg3_table:
        [spf, opf, opcode] = bmi_arg3_table[inst.mnem]
        [dst, src1, src2] = inst.args
        if inst.mnem in bmi_arg3_reversed:
            [src2, src1] = [src1, src2]
        return vex(w, dst, src1, opf, src2, 0, spf) + [opcode] + mod_rm_sib(dst, src1)
    elif inst.mnem in jump_table:
        opcode = jump_table[inst.mnem]
        [src] = inst.args
        # XXX Since we don't know how far or in what direction we're jumping,
        # punt and use disp32. We'll fill the offset in later.
        return [0x0F, opcode, Relocation(src, 4)]
    elif inst.mnem in ['jmp', 'call']:
        [src] = inst.args
        [opcode, sub_opcode] = {'jmp': [0xE9, 4], 'call': [0xE8, 2]}[inst.mnem]
        if isinstance(src, Label):
            return [opcode, Relocation(src, 4)]
        else:
            return rex(0, 0, src) + [0xFF] + mod_rm_sib(sub_opcode, src)
    elif inst.mnem in setcc_table:
        opcode = setcc_table[inst.mnem]
        [dst] = inst.args
        # Force REX since ah/bh etc. are used instead of sil etc. without it
        force = isinstance(dst, GPReg) and dst.index & 0xC == 4
        return rex(0, 0, dst, force=force) + [0x0F, opcode] + mod_rm_sib(0, dst)
    assert False, inst.mnem

# Create raw code and data sections, along with a list of local/global/external
# labels, suitable for passing directly to elf.create_elf_file.
def build(insts):
    code = []
    data = []
    data_sym_idx = 0

    local_labels = []
    global_labels = []
    extern_labels = []
    data_labels = {}
    relocations = []
    for inst in insts:
        if isinstance(inst, LocalLabel):
            local_labels.append((inst.name, 'code', len(code)))
        elif isinstance(inst, GlobalLabel):
            global_labels.append((inst.name, 'code', len(code)))
        else:
            for byte in assemble_inst(inst):
                if not isinstance(byte, Relocation):
                    code.append(byte)
                    continue

                target = byte.label
                # If the relocation is to an external symbol, pass it on
                if isinstance(target, (ExternLabel, Data)):
                    # Also add bytes to the data section if this is a Data
                    # object. We only do this once for a given object.
                    if isinstance(target, Data):
                        if target not in data_labels:
                            data_sym = 'data$%s' % data_sym_idx
                            data_labels[target] = data_sym
                            data_sym_idx += 1
                            # Handle alignment
                            padding = -len(data) & (target.alignment - 1)
                            data.extend([0] * padding)
                            assert not len(data) & (target.alignment - 1)
                            # Add the data to the data section
                            global_labels.append((data_sym, 'data', len(data)))
                            data.extend(target.data_bytes)
                        target = data_labels[target]
                    else:
                        target = target.name

                    assert byte.size == 4
                    extern_labels.append((target, 'code', len(code)))
                    # HACKish: assume the relocation is at the end of an
                    # instruction. Since the PC will be 4 bytes after the
                    # end of this value when the instruction executes, and
                    # the linker will calculate the offset from the
                    # beginning of the value, put an offset of -4 here that
                    # the linker will add in. We also add the offset
                    # from this relocation.
                    code += pack_bits(byte.offset - 4, 32)
                else:
                    relocations += [[byte, len(code)]]
                    code += [0] * byte.size

    # Patch local relocations, now that we've built the code section and know
    # the offsets of local/global labels
    if relocations:
        label_dict = {label: [section, address]
                for labels in [local_labels, global_labels]
                for [label, section, address] in labels}
        assert len(label_dict) == (len(local_labels) + len(global_labels))
        new_bytes = []
        last_offset = 0
        for [rel, offset] in relocations:
            new_bytes += code[last_offset:offset]
            # XXX only 4-byte for now
            assert rel.size == 4
            [section, rel_offset] = label_dict[rel.label.name]
            disp = rel_offset - offset - rel.size
            new_bytes += pack_bits(disp, 32)
            last_offset = offset + rel.size
        code = new_bytes + code[last_offset:]

    return [code, data, global_labels, extern_labels]
