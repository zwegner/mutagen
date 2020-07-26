# Hacky relative import function from tests/py/utils.py
def import_path(path):
    import os
    import sys
    base = os.path.dirname(sys.argv[0])
    path = os.path.abspath(base + '/' + path)
    path, file = os.path.split(path)
    file, ext = os.path.splitext(file)
    sys.path = [path] + sys.path
    module = __import__(file)
    sys.path.pop(0)
    return module

asm = import_path('../../bootstrap/asm.py')
elf = import_path('../../bootstrap/elf.py')

# RKISS algorithm
def gen_rand_64():
    def rol(x, y):
        return (x << y) | (x >> 64 - y)
    def int64(x):
        return x & (1 << 64) - 1

    rand_state = [0x8C84A911159F4017, 0x062C0B602809C02E, 0xA48B831518DEA5D7,
            0x55AB3636D17F3AD3]
    while True:
        [a, b, c, d] = rand_state
        e = a - rol(b, 7)
        a = b ^ rol(c, 13)
        b = c + rol(d, 37)
        c = d + e
        d = e + a
        rand_state = [int64(a), int64(b), int64(c), int64(d)]
        yield int64(d)

# Basic coroutine type thingy, to somewhat match the Mutagen effect-based version
RAND = iter(gen_rand_64())
def rand_select(l):
    i = next(RAND)
    return l[i % len(l)]

regs = list(range(16))
bases = list(range(-1, 16))
scales = [0, 1, 2, 4, 8]
indices = [r for r in range(16) if r != 4] # index can't be RSP
# Our IR can't properly print unsigned integers without a bunch of work,
# as they appear in the objdump output. So no negative numbers for now.
imm_bytes = [0, 1, 7, 37]
imms = imm_bytes + [0xFF, 0x100, 0xFFFFFF]

labels = [asm.LocalLabel(l) for l in ['_start', '_end']]
insts = [asm.GlobalLabel('_start')]

# Generate a bunch of instructions. We loop over all forms of all instructions,
# to make sure each form is tested, and we test 5 batches of random arguments
# for each form
for spec in asm.INST_SPECS.values():
    for form in spec.forms:
        for j in range(5):
            args = []
            for arg in form:
                if isinstance(arg, asm.ASMObj):
                    args.append(arg)
                    continue
                assert isinstance(arg, tuple), arg
                [arg_type, size] = arg
                if arg_type == asm.GPReg:
                    arg = rand_select(regs)
                    arg = asm.GPReg(arg, size=size)
                elif arg_type == asm.BaseAddress:
                    base = rand_select(bases)
                    if base == -1:
                        [scale, index] = [0, 0]
                    else:
                        scale = rand_select(scales)
                        index = rand_select(indices) if scale else 0
                    disp = rand_select(imms)
                    arg = asm.Address(base, scale, index, disp, size=size)
                elif arg_type is asm.Immediate:
                    arg = rand_select(range(1 << size))
                    arg = asm.Immediate(arg, size=size)
                elif arg_type == asm.Label:
                    arg = rand_select(labels)
                elif arg_type == asm.VecReg:
                    arg = rand_select(regs)
                    arg = asm.VecReg(arg, size=size)
                else:
                    assert False, arg
                args = args + [arg]
            insts.append(asm.Instruction(spec.inst, *args))

# Add the end label, plus an extra instruction, so objdump still prints it
insts = insts + [asm.GlobalLabel('_end'), asm.Instruction('ret')]

elf_file = elf.create_elf_file(*asm.build(insts))
with open('elfout.o', 'wb') as f:
    f.write(bytes(elf_file))

# Print out our interpretation of the instructions, so it can be matched
# against objdump
for inst in insts:
    if isinstance(inst, asm.Label):
        print('{}:'.format(inst.name))
    else:
        print('    {}'.format(inst))
