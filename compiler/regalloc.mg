import asm
import elf

class BasicBlock(insts):
    def gen_insts(self):
        # Generate the set of live vars at each point in the instruction list
        # We go backwards so we know when things are actually used.
        live_set = set() # XXX outs
        live_sets = []
        for [i, inst] in reversed(list(enumerate(self.insts))):
            # Add the argument from the last iteration, since we're interested
            # in what's used after the instruction
            live_sets = live_sets + [live_set]

            if inst[0] != 'literal':
                for arg in slice(inst, 1, None):
                    live_set = live_set | {arg}

            # Make sure we aren't live before we exist
            live_set = live_set - {i}
        # Now put the list back in forward order
        live_sets = list(reversed(live_sets))

        # Now make a run through the instructions. Since we're assuming
        # spill/fill has been taken care of already, this can be done linearly.
        free_regs = set(range(16))
        insts = []
        reg_assns = {}
        for [i, [inst, live_set]] in enumerate(zip(self.insts, live_sets)):
            [inst, args] = [inst[0], slice(inst, 1, None)]
            # Special literal opcode: just a placeholder so we can differentiate
            # instruction indices and just literal ints/strs/whatever.
            if inst == 'literal':
                reg_assns = reg_assns + {i: args[0]}
            elif args:
                arg_regs = [reg_assns[i] for i in args]

                # Return any now-unused sources to the free set.
                for [arg, dest] in slice(list(zip(args, arg_regs)), 1, None):
                    if isinstance(dest, asm.Register) and arg not in live_set:
                        free_regs = {dest.index} | free_regs

                if (asm.is_destructive(inst) and
                    isinstance(arg_regs[0], asm.Register)):
                    # See if we can clobber the register. If not, copy it
                    # into another register and change the assignment so
                    # later ops can see it.
                    if args[0] in live_set:
                        [free_regs, new_reg] = free_regs.pop()
                        insts = insts + [asm.Instruction('mov64',
                            asm.Register(new_reg), arg_regs[0])]
                        reg_assns = reg_assns + {args[0]: asm.Register(new_reg)}

                    reg = arg_regs[0].index
                else:
                    # Non-destructive ops need a register assignment for their
                    # implicit destination
                    [free_regs, reg] = free_regs.pop()
                    arg_regs = [asm.Register(reg)] + arg_regs

                # And now return the destination of the instruction to the
                # free registers, although this only happens when the result
                # is never used...
                if i not in live_set:
                    free_regs =  {arg_regs[0].index} | free_regs

                reg_assns = reg_assns + {i: asm.Register(reg)}

                insts = insts + [asm.Instruction(inst, *arg_regs)]
            else:
                insts = insts + [asm.Instruction(inst)]

        return insts

b = BasicBlock([
    ['literal', 11],
    ['literal', 22],
    ['mov64', 0],
    ['mov64', 1],
    ['mulx64', 2, 2],
    ['mulx64', 3, 3],
    ['mulx64', 4, 4],
    ['mulx64', 2, 2],
    ['mulx64', 5, 5],
    ['add64', 2, 2],
    ['add64', 3, 3],
    ['add64', 4, 4],
    ['add64', 2, 2],
    ['add64', 5, 5],
    ['ret'],
    ])

insts = b.gen_insts()

for inst in insts:
    if isinstance(inst, asm.Label):
        print('{}:'.format(inst.name))
    else:
        print('    {}'.format(inst))
