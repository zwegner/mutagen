import asm from '../../compiler/asm.mg'
import elf from '../../compiler/elf.mg'

# RKISS algorithm
def gen_rand_64(n):
    def rol(x, y):
        return (x << y) | (x >> 64 - y)
    def int64(x):
        return x & (1 << 64) - 1

    rand_state = [0x8C84A911159F4017, 0x062C0B602809C02E, 0xA48B831518DEA5D7,
            0x55AB3636D17F3AD3]
    for i in range(n):
        [a, b, c, d] = rand_state
        e = a - rol(b, 7)
        a = b ^ rol(c, 13)
        b = c + rol(d, 37)
        c = d + e
        d = e + a
        rand_state = [int64(a), int64(b), int64(c), int64(d)]
        yield int64(d)

def rand_select(l, i):
    return [l[i % len(l)], i // len(l)]

regs = [asm.Register(i) for i in range(16)]
bases = list(range(-1, 16))
scales = [1, 2, 4, 8]
indices = list(range(4)) + list(range(5, 16)) # index can't be RSP
# Our IR can't properly print unsigned integers without a bunch of work,
# as they appear in the objdump output. So no negative numbers for now.
imms = [0, 1, 7, 37, 0xFF, 0x100, 0xFFFFFF]

labels = [asm.Label(l, False) for l in ['_start', '_end']]

# Generate a bunch of random instructions. Should make sure this hits every
# instruction somehow (random.shuffle?). This is also a good case for figuring
# out some way to thread a stream of random numbers through for every place
# that needs one.
inst_specs = list(asm.get_inst_specs())
insts = [asm.Label('_start', True)]
for rand in gen_rand_64(1000):
    [inst_spec, rand] = rand_select(inst_specs, rand)
    args = []
    for arg_spec in slice(inst_spec, 1, None):
        [arg_type, rand] = rand_select(arg_spec, rand)
        if arg_type == 'r':
            [arg, rand] = rand_select(regs, rand)
        elif arg_type == 'a':
            [base, rand] = rand_select(bases, rand)
            if base == -1:
                [scale, index] = [0, 0]
            else:
                [scale, rand] = rand_select(scales, rand)
                [index, rand] = rand_select(indices, rand)
            [disp, rand] = rand_select(imms, rand)
            arg = asm.Address(base, scale, index, disp)
        elif arg_type == 'i':
            [arg, rand] = rand_select(imms, rand)
        elif arg_type == 'l':
            [arg, rand] = rand_select(labels, rand)
        else:
            assert False
        args = args + [arg]
    insts = insts + [asm.Instruction(inst_spec[0], *args)]

# Add the end label, plus an extra instruction, so objdump still prints it
insts = insts + [asm.Label('_end', True), asm.Instruction('ret')]

elf_file = elf.create_elf_file(*asm.build(insts))
write_binary_file('elfout.o', elf_file)

# Print out our interpretation of the instructions, so it can be matched
# against objdump
for inst in insts:
    if isinstance(inst, asm.Label):
        print('{}:'.format(inst.name))
    else:
        print('    {}'.format(inst))
