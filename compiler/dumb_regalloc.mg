import regalloc
import elf
# XXX work around weird importing behavior
asm = regalloc.asm

class Function(parameters, blocks):
    pass

class BasicBlock(name, phis, insts):
    pass

param_regs = [7, 6, 2, 1]

def gen_insts(name, fn):
    stack_slot = -8
    insts = []

    [rsp, rbp, rax] = [asm.Register(4), asm.Register(5), asm.Register(0)]
    all_registers = set(range(16)) - set(param_regs) - {rsp.index, rbp.index}

    phi_reg_assns = reg_assns = {i: {} for i in range(len(fn.blocks))}

    # Assign phi stack slots. Since multiple phis can read the same value, we
    # need to maintain a list of slot assignments, not just one.
    for [block_id, block] in enumerate(fn.blocks):
        for [inst_id, inst] in enumerate(block.insts):
            [opcode, args] = [inst[0], inst[1:]]
            if opcode == 'phi':
                # Assign a slot
                dest = asm.Address(rbp.index, 0, 0, stack_slot)
                stack_slot = stack_slot - 8
                assert inst_id not in reg_assns[block_id]
                reg_assns = reg_assns <- [block_id][inst_id] = dest

                # Add it to the phi list of each phi argument
                for [src_block, src_inst] in args:
                    if src_inst in phi_reg_assns[src_block]:
                        old = phi_reg_assns[src_block][src_inst]
                    else:
                        old = []
                    phi_reg_assns = phi_reg_assns <- [src_block][src_inst] = old + [dest]
            # Phis are always at the beginning
            else:
                break

    for [block_id, block] in enumerate(fn.blocks):
        insts = insts + [asm.LocalLabel(block.name)]
        for [inst_id, inst] in enumerate(block.insts):
            [opcode, args] = [inst[0], inst[1:]]
            # Special literal opcode: just a placeholder so we can differentiate
            # instruction indices and just literal ints/strs/whatever.
            if opcode == 'literal':
                reg_assns = reg_assns <- [block_id][inst_id] = args[0]
            # We assigned stack slots for phis above, but we delegate the responsibility
            # of writing to the slot to the source instructions in all predecessor blocks.
            # Given that, we need to check if this phi is a source for another phi.
            elif opcode == 'phi':
                if inst_id in phi_reg_assns[block_id]:
                    [_, reg] = all_registers.pop()
                    reg = asm.Register(reg)
                    # For each phi that reads this phi value, we need two movs, since
                    # x86 doesn't have mem->mem copies.
                    for phi in phi_reg_assns[block_id][inst_id]:
                        insts = insts + [asm.Instruction('mov64', reg,
                                reg_assns[block_id][inst_id]),
                            asm.Instruction('mov64', phi, reg)]
            # Return: optionally move the argument into eax and jump to the exit block.
            # We jump since we need to take care of the stack but we don't yet know how
            # much space we allocate.
            elif opcode == 'ret':
                if args:
                    arg = reg_assns[block_id][args[0]]
                    insts = insts + [asm.Instruction('mov64', asm.Register(0), arg)]
                insts = insts + [asm.Instruction('jmp', asm.LocalLabel('exit'))]
            # Jump instructions
            elif asm.is_jump_op(opcode):
                # Make sure only the last instruction is a control flow op
                assert inst_id == len(block.insts) - 1
                # XXX take a flags argument, and make sure it's from the latest
                # flags-writing instruction
                [dest_block_id] = args
                if isinstance(dest_block_id, asm.Label):
                    dest_block_id = dest_block_id.name
                dest = asm.LocalLabel(dest_block_id)
                insts = insts + [asm.Instruction(opcode, dest)]
            # 0-operand instructions: just add it here and don't bother with registers
            elif not args:
                insts = insts + [asm.Instruction(opcode)]
            # All other opcode types: we 
            else:
                # Parameter: load a value from the proper register given standard C ABI
                if opcode == 'parameter':
                    [arg] = args
                    reg = asm.Register(param_regs[arg])
                elif opcode == 'call':
                    [fn, args] = [args[0], args[1:]]
                    assert len(args) <= len(param_regs)
                    # Load all arguments from the corresponding stack locations
                    for [arg, reg] in zip(args, param_regs):
                        arg = reg_assns[block_id][arg]
                        assert isinstance(arg, asm.Address)
                        insts = insts + [asm.Instruction('mov64', asm.Register(reg), arg)]
                    insts = insts + [asm.Instruction(opcode, fn)]
                    # C ABI dictates the return value from function is in rax
                    reg = rax
                elif args:
                    # Load all arguments from the corresponding stack locations
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

                    # Assign a destination to 3-operand instructions
                    if asm.needs_register(opcode) and not (asm.is_destructive_op(opcode) and
                            isinstance(arg_regs[0], asm.Register)):
                        [free_regs, reg] = free_regs.pop()
                        arg_regs = [asm.Register(reg)] + arg_regs
                    reg = arg_regs[0]

                    # Add the instruction as well as a store into our stack slot
                    insts = insts + [asm.Instruction(opcode, *arg_regs)]

                # Store the result on the stack
                if asm.needs_register(opcode):
                    # Assign a stack slot
                    dest = asm.Address(rbp.index, 0, 0, stack_slot)
                    stack_slot = stack_slot - 8
                    reg_assns = reg_assns <- [block_id][inst_id] = dest

                    insts = insts + [asm.Instruction('mov64', dest, reg)]
                    # ...and again for any phi that references it
                    if inst_id in phi_reg_assns[block_id]:
                        for phi in phi_reg_assns[block_id][inst_id]:
                            insts = insts + [asm.Instruction('mov64', phi, reg)]
                else:
                    assert inst_id not in phi_reg_assns[block_id]

    # Now that we know the total amount of stack space allocated, add a preamble and postamble
    stack_size = -8 - stack_slot
    insts = [asm.GlobalLabel(name),
        asm.Instruction('push64', rbp),
        asm.Instruction('mov64', rbp, rsp),
        asm.Instruction('sub64', rsp, stack_size)
    ] + insts + [
        asm.LocalLabel('exit'),
        asm.Instruction('add64', rsp, stack_size),
        asm.Instruction('pop64', rbp),
        asm.Instruction('ret')
    ]

    return insts

def print_insts(insts):
    for inst in insts:
        if isinstance(inst, asm.Label):
            print('{}:'.format(inst.name))
        else:
            print('    {}'.format(inst))

def export_functions(file, fns):
    all_insts = []
    for [name, fn] in fns:
        insts = gen_insts(name, fn)
        print_insts(insts)
        all_insts = all_insts + insts
    elf_file = elf.create_elf_file(*asm.build(all_insts))
    write_binary_file(file, elf_file)
