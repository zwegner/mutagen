import asm
import elf

class BasicBlock(insts):
    pass

param_regs = [7, 6, 2]

def gen_insts(name, blocks):
    stack_slot = -8
    insts = []

    [rsp, rbp] = [asm.Register(4), asm.Register(5)]
    all_registers = set(range(16)) - set(param_regs) - {rsp.index, rbp.index}

    phi_reg_assns = reg_assns = {i: {} for i in range(len(blocks))}

    # Assign phi stack slots. Since multiple phis can read the same value, we
    # need to maintain a list of slot assignments, not just one.
    for [block_id, block] in enumerate(blocks):
        for [i, inst] in enumerate(block.insts):
            [opcode, args] = [inst[0], inst[1:]]
            if opcode == 'phi':
                # Assign a slot
                dest = asm.Address(rbp.index, 0, 0, stack_slot)
                stack_slot = stack_slot - 8
                assert i not in reg_assns[block_id]
                reg_assns = reg_assns + {block_id: reg_assns[block_id] + {i: dest}}

                # Add it to the phi list of each phi argument
                for [src_block, src_inst] in args:
                    if src_inst in phi_reg_assns[src_block]:
                        old = phi_reg_assns[src_block][src_inst]
                    else:
                        old = []
                    phi_reg_assns = phi_reg_assns + {src_block: phi_reg_assns[src_block] +
                        {src_inst: old + [dest]}}

    for [block_id, block] in enumerate(blocks):
        insts = insts + [asm.Label('block{}'.format(block_id), False)]
        for [i, inst] in enumerate(block.insts):
            [opcode, args] = [inst[0], inst[1:]]
            # Special literal opcode: just a placeholder so we can differentiate
            # instruction indices and just literal ints/strs/whatever.
            if opcode == 'literal':
                reg_assns = reg_assns + {block_id: reg_assns[block_id] + {i: args[0]}}
            # We already dealt with phis above
            elif opcode == 'phi':
                pass
            # Hacky dup pseudo-op: used to simplify store handling. Just copy a register assignment.
            elif opcode == 'dup':
                [arg] = args
                reg_assns = reg_assns + {block_id: reg_assns[block_id] + {i: reg_assns[block_id][arg]}}
            # Return: optionally move the argument into eax and jump to the exit block.
            # We jump since we need to take care of the stack but we don't yet know how
            # much space we allocate.
            elif opcode == 'ret':
                if args:
                    [arg] = args
                    arg = reg_assns[block_id][arg]
                    insts = insts + [asm.Instruction('mov64', asm.Register(0), arg)]
                insts = insts + [asm.Instruction('jmp', asm.Label('exit', False))]
            # Parameter: load a value from the proper register given standard C ABI
            elif opcode == 'parameter':
                [arg] = args
                arg = asm.Register(param_regs[arg])
                if i in phi_reg_assns[block_id]:
                    for phi in phi_reg_assns[block_id][i]:
                        insts = insts + [asm.Instruction('mov64', phi, arg)]
                else:
                    dest = asm.Address(rbp.index, 0, 0, stack_slot)
                    stack_slot = stack_slot - 8
                    reg_assns = reg_assns + {block_id: reg_assns[block_id] + {i: dest}}
                    insts = insts + [asm.Instruction('mov64', dest, arg)]
            elif asm.is_jump_op(opcode):
                # Make sure only the last instruction is a control flow op
                assert i == len(block.insts) - 1
                # XXX take a flags argument, and make sure it's from the latest
                # flags-writing instruction
                [dest_block_id] = args
                dest = asm.Label('block{}'.format(dest_block_id), False)
                insts = insts + [asm.Instruction(opcode, dest)]
            elif args:
                # Load ll arguments from the corresponding stack locations
                free_regs = all_registers
                arg_regs = []
                for arg in args:
                    arg = reg_assns[block_id][arg]
                    if isinstance(arg, asm.Address):
                        [free_regs, reg] = free_regs.pop()
                        reg = asm.Register(reg)
                        arg_regs = arg_regs + [reg]
                        insts = insts + [asm.Instruction('mov64', reg, arg)]
                    else:
                        arg_regs = arg_regs + [arg]

                # Assign a stack slot
                dest = asm.Address(rbp.index, 0, 0, stack_slot)
                stack_slot = stack_slot - 8
                reg_assns = reg_assns + {block_id: reg_assns[block_id] + {i: dest}}

                # Assign a destination to 3-operand instructions
                destructive = (asm.is_destructive_op(opcode) and
                    isinstance(arg_regs[0], asm.Register))
                if not destructive and asm.needs_register(opcode):
                    [free_regs, reg] = free_regs.pop()
                    reg = asm.Register(reg)
                    arg_regs = [reg] + arg_regs
                else:
                    reg = arg_regs[0]

                # Add the instruction as well as a store into our stack slot
                insts = insts + [asm.Instruction(opcode, *arg_regs)]

                # Store the result on the stack
                if asm.needs_register(opcode):
                    insts = insts + [asm.Instruction('mov64', dest, reg)]

                # ...and again for any phi that references it
                if i in phi_reg_assns[block_id]:
                    for phi in phi_reg_assns[block_id][i]:
                        insts = insts + [asm.Instruction('mov64', phi, reg)]
            else:
                insts = insts + [asm.Instruction(opcode)]

    # Now that we know the total amount of stack space allocated, add a preamble and postamble
    stack_size = -8 - stack_slot
    insts = [asm.Label(name, True),
        asm.Instruction('push64', rbp),
        asm.Instruction('mov64', rbp, rsp),
        asm.Instruction('sub64', rsp, stack_size)
    ] + insts + [
        asm.Label('exit', False),
        asm.Instruction('add64', rsp, stack_size),
        asm.Instruction('pop64', rbp),
        asm.Instruction('ret')
    ]

    return insts

blocks = [
    BasicBlock([
        ['parameter', 0],
        ['literal', 0],
        ['mov64', 1],
    ]),
    BasicBlock([
        ['phi', [0, 0], [1, 4]],
        ['phi', [0, 2], [1, 3]],
        ['literal', 1],
        ['add64', 1, 0],
        ['sub64', 0, 2],
        ['jnz', 1],
    ]),
    BasicBlock([
        ['phi', [1, 3]],
        ['ret', 0],
    ]),
]

def print_insts(insts):
    for inst in insts:
        if isinstance(inst, asm.Label):
            print('{}:'.format(inst.name))
        else:
            print('    {}'.format(inst))

def export_function(file, name, blocks):
    insts = gen_insts(name, blocks)
    elf_file = elf.create_elf_file(*asm.build(insts))
    write_binary_file(file, elf_file)