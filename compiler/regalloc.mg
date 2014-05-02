import asm
import elf

class BasicBlock(insts):
    def get_live_sets(self, outs):
        # Generate the set of live vars at each point in the instruction list
        # We go backwards so we know when things are actually used.
        live_set = outs
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

        # Be sure to put the list back in forward order
        return list(reversed(live_sets))

def gen_insts(blocks):
    block_insts = []
    block_outs = {block_id: {} for [block_id, block] in enumerate(blocks)}
    # Go through blocks in reverse order. Right now, assuming no loops, and
    # the blocks are in physical order, so are an implicit topographical ordering.
    # This means we can go through backwards and assign registers to phis, and
    # then in predecessor blocks make sure all block outs are pre-assigned to
    # those registers.
    for [block_id, block] in reversed(list(enumerate(blocks))):
        insts = [asm.Label('block{}'.format(block_id), False)]

        live_sets = block.get_live_sets(set(block_outs[block_id].keys()))

        # Now make a run through the instructions. Since we're assuming
        # spill/fill has been taken care of already, this can be done linearly.
        free_regs = set(range(16))
        reg_assns = {}
        for [i, [inst, live_set]] in enumerate(zip(block.insts, live_sets)):
            [opcode, args] = [inst[0], slice(inst, 1, None)]
            # Special literal opcode: just a placeholder so we can differentiate
            # instruction indices and just literal ints/strs/whatever.
            if opcode == 'literal':
                reg_assns = reg_assns + {i: args[0]}
            elif opcode == 'phi':
                # Assign a register
                [free_regs, reg] = free_regs.pop()
                reg_assns = reg_assns + {i: asm.Register(reg)}
                # Go through all operands, which consist of a block and an
                # operand within that block, and make sure we add them to the
                # live outs of the proper block, and mark the assigned register.
                for [src_block_id, src_op] in args:
                    outs = block_outs[src_block_id] + {src_op: reg}
                    block_outs = block_outs + {src_block_id: outs}
            elif asm.is_control_flow_op(opcode):
                # Make sure only the last instruction is a control flow op
                assert i == len(block.insts) - 1
                # XXX take a flags argument, and make sure it's from the latest
                # flags-writing instruction
                [dest_block_id] = args
                dest = asm.Label('block{}'.format(dest_block_id), False)
                insts = insts + [asm.Instruction(opcode, dest)]
            elif args:
                arg_regs = [reg_assns[i] for i in args]

                # Return any now-unused sources to the free set.
                for [arg, dest] in slice(list(zip(args, arg_regs)), 1, None):
                    if isinstance(dest, asm.Register) and arg not in live_set:
                        free_regs = {dest.index} | free_regs

                if (asm.is_destructive_op(opcode) and
                    isinstance(arg_regs[0], asm.Register)):
                    # See if we can clobber the register. If not, copy it
                    # into another register and change the assignment so
                    # later ops can see it.
                    # XXX This can be done in two ways, but moreover this
                    # should probably done during scheduling with spill/fill.
                    # This also needs to interact with coalescing, when we have that.
                    if args[0] in live_set:
                        [free_regs, reg] = free_regs.pop()
                        insts = insts + [asm.Instruction('mov64',
                            asm.Register(reg), arg_regs[0])]
                    else:
                        reg = arg_regs[0].index
                elif asm.needs_register(opcode):
                    # Non-destructive ops need a register assignment for their
                    # implicit destination
                    # First, check if this operand is a live out of the block,
                    # in which case the register is already assigned.
                    # XXX only valid now with no loops
                    if i in block_outs[block_id]:
                        reg = block_outs[block_id][i]
                    else:
                        [free_regs, reg] = free_regs.pop()
                    arg_regs = [asm.Register(reg)] + arg_regs

                reg_assns = reg_assns + {i: asm.Register(reg)}

                # And now return the destination of the instruction to the
                # free registers, although this only happens when the result
                # is never used...
                if i not in live_set:
                    free_regs =  {arg_regs[0].index} | free_regs

                insts = insts + [asm.Instruction(opcode, *arg_regs)]
            else:
                insts = insts + [asm.Instruction(opcode)]

        block_insts = block_insts + [insts]

    return sum(reversed(block_insts), [])

blocks = [
    BasicBlock([
        ['literal', 11],
        ['mov64', 0],
        ['cmp64', 1, 0],
        ['je', 2],
    ]),
    BasicBlock([
        ['literal', 22],
        ['mov64', 0],
    ]),
    BasicBlock([
        ['phi', [0, 1], [1, 1]],
        ['literal', 33],
        ['add64', 0, 1],
        ['ret'],
    ]),
]

insts = gen_insts(blocks)

for inst in insts:
    if isinstance(inst, asm.Label):
        print('{}:'.format(inst.name))
    else:
        print('    {}'.format(inst))
