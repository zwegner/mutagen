import random

import asm from '../../compiler/asm.mg'
import elf from '../../compiler/elf.mg'

def rand_select(l):
    r = (perform random.GetRandomInt64Effect())
    return l[r % len(l)]

regs = [asm.Register(i) for i in range(16)]
bases = list(range(-1, 16))
scales = [0, 1, 2, 4, 8]
indices = list(range(4)) + list(range(5, 16)) # index can't be RSP
# Our IR can't properly print unsigned integers without a bunch of work,
# as they appear in the objdump output. So no negative numbers for now.
imms = [0, 1, 7, 37, 0xFF, 0x100, 0xFFFFFF]

labels = [asm.LocalLabel(l) for l in ['_start', '_end']]

inst_specs = list(asm.get_inst_specs())

# Generate a bunch of random instructions. Should make sure this hits every
# instruction somehow (random.shuffle?). This is a function so we can provide
# randomness by handling effects in random.handle_randomness().
def gen_random_insts():
    insts = [asm.GlobalLabel('_start')]
    for i in range(500):
        inst_spec = rand_select(inst_specs)
        args = []
        for arg_spec in inst_spec[1:]:
            arg_type = rand_select(arg_spec)
            if arg_type == 'r':
                arg = rand_select(regs)
            elif arg_type == 'a':
                base = rand_select(bases)
                if base == -1:
                    [scale, index] = [0, 0]
                else:
                    scale = rand_select(scales)
                    index = rand_select(indices) if scale else 0
                disp = rand_select(imms)
                arg = asm.Address(base, scale, index, disp)
            elif arg_type == 'i':
                arg = rand_select(imms)
            elif arg_type == 'l':
                arg = rand_select(labels)
            else:
                assert False
            args = args + [arg]
        insts = insts + [asm.Instruction(inst_spec[0], *args)]
    return insts

insts = random.handle_randomness(gen_random_insts)

# Add the end label, plus an extra instruction, so objdump still prints it
insts = insts + [asm.GlobalLabel('_end'), asm.Instruction('ret')]

elf_file = elf.create_elf_file(*asm.build(insts))
write_binary_file('elfout.o', elf_file)

# Print out our interpretation of the instructions, so it can be matched
# against objdump
for inst in insts:
    if isinstance(inst, asm.Label):
        print('{}:'.format(inst.name))
    else:
        print('    {}'.format(inst))
