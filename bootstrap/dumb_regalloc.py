import asm
import elf
import lir

# SysV AMD64 calling convention
PARAM_REGS = [7, 6, 2, 1] # rdi, rsi, rdx, rcx
RETURN_REGS = [0] # rax

def get_arg_reg(arg, reg_assns, free_regs, insts, force_move=False):
    is_lir_node = isinstance(arg, lir.Node)
    if is_lir_node or force_move:
        # Anything that isn't a LIR node, we assume is a literal that asm.py can handle
        if is_lir_node:
            arg = reg_assns[arg]
        reg = free_regs.pop(0)
        reg = asm.Register(reg)
        insts.append(asm.Instruction('mov64', reg, arg))
        return reg
    else:
        return arg

def allocate_registers(name, fn):
    stack_slot = -8
    insts = []

    [rsp, rbp, rax] = [asm.Register(4), asm.Register(5), asm.Register(0)]
    all_registers = list(set(range(16)) - set(PARAM_REGS) - {rsp.index, rbp.index})

    reg_assns = {}
    phi_reg_assns = {}

    # Assign phi stack slots. Since multiple phis can read the same value, we
    # need to maintain a list of slot assignments, not just one. Each value that
    # is used by phis will later store its value in each of those registers.
    for block in fn.blocks:
        for phi in block.phis:
            # Assign a slot
            dest = asm.Address(rbp.index, 0, 0, stack_slot)
            stack_slot -= 8
            assert phi not in reg_assns
            reg_assns[phi] = dest

            # Add it to the phi list of each phi argument
            for src_node in phi.args:
                if src_node not in phi_reg_assns:
                    phi_reg_assns[src_node] = []
                phi_reg_assns[src_node] += [dest]

    for block in fn.blocks:
        insts.append(asm.LocalLabel(block.name))

        for phi in block.phis:
            # We assigned stack slots for phis above, but we delegate the responsibility
            # of writing to the slot to the source instructions in all predecessor blocks.
            # Given that, we need to check if this phi is a source for another phi.
            if phi in phi_reg_assns:
                reg = asm.Register(all_registers[0])
                # For each phi that reads this phi value, we need two movs, since
                # x86 doesn't have mem->mem copies.
                for phi_dest in phi_reg_assns[phi]:
                    insts += [
                            asm.Instruction('mov64', reg, reg_assns[phi]),
                            asm.Instruction('mov64', phi_dest, reg)]

        for [inst_id, inst] in enumerate(block.insts):
            [opcode, args] = [inst.opcode, inst.args]

            # Return: optionally move the argument into eax and jump to the exit block.
            # We jump since we need to take care of the stack but we don't yet know how
            # much space we allocate.
            if opcode == 'ret':
                if args:
                    assert len(args) == 1
                    # Put the return value in a register. Since the return value isn't actually
                    # part of the ret instruction, be sure to force a move to rax in case the
                    # return value is a literal.
                    arg = get_arg_reg(args[0], reg_assns, RETURN_REGS.copy(), insts, force_move=True)
                insts.append(asm.Instruction('jmp', asm.LocalLabel('exit')))
            # Jump instructions
            elif asm.is_jump_op(opcode):
                # Make sure only the last instruction is a control flow op
                # XXX not true at the moment, we have two jumps at the end of every branching block
                #assert inst_id == len(block.insts) - 1

                # XXX take a flags argument, and make sure it's from the latest
                # flags-writing instruction
                [dest_block_id] = args
                assert isinstance(dest_block_id, asm.Label), str(dest_block_id)
                dest_block_id = dest_block_id.name
                dest = asm.LocalLabel(dest_block_id)
                insts.append(asm.Instruction(opcode, dest))
            # All other opcode types
            else:
                # Parameter: load a value from the proper register given by the ABI
                if opcode == 'parameter':
                    [arg] = args
                    reg = asm.Register(PARAM_REGS[arg])
                elif opcode == 'call':
                    [called_fn, args] = [args[0], args[1:]]
                    # Load all arguments from the corresponding stack locations
                    assert len(args) <= len(PARAM_REGS)
                    free_regs = PARAM_REGS.copy()
                    for arg in args:
                        _ = get_arg_reg(arg, reg_assns, free_regs, insts, force_move=True)

                    insts.append(asm.Instruction(opcode, called_fn))
                    # Use the ABI-specified register for pulling out the return value from the call
                    reg = asm.Register(RETURN_REGS[0])
                # Zero out destination for setcc instructions, since they only set the lower 8 bits.
                elif opcode in asm.setcc_table:
                    dst = asm.Register(all_registers[0])
                    insts += [
                        # Use mov r, 0 rather than xor r, r since the latter clears flags...
                        asm.Instruction('mov64', dst, 0),
                        asm.Instruction(opcode, dst)
                    ]
                elif args:
                    # Load all arguments from the corresponding stack locations.
                    free_regs = all_registers.copy()
                    arg_regs = []
                    for arg in args:
                        arg = get_arg_reg(arg, reg_assns, free_regs, insts)
                        arg_regs.append(arg)

                    # Assign a destination to 3-operand instructions
                    if asm.needs_register(opcode) and not (asm.is_destructive_op(opcode) and
                            isinstance(arg_regs[0], asm.Register)):
                        reg = free_regs.pop(0)
                        arg_regs = [asm.Register(reg)] + arg_regs
                    reg = arg_regs[0]

                    # Add the instruction as well as a store into our stack slot
                    insts.append(asm.Instruction(opcode, *arg_regs))

                # Store the result on the stack
                if asm.needs_register(opcode):
                    # Assign a stack slot
                    dest = asm.Address(rbp.index, 0, 0, stack_slot)
                    stack_slot = stack_slot - 8
                    reg_assns[inst] = dest

                    insts.append(asm.Instruction('mov64', dest, reg))
                    # ...and again for any phi that references it
                    if inst in phi_reg_assns:
                        for phi in phi_reg_assns[inst]:
                            insts.append(asm.Instruction('mov64', phi, reg))
                else:
                    assert inst not in phi_reg_assns

    # Round up to the nearest multiple of 16. Only add 7 since -stack_slot is 8 more
    # the actual stack size needed.
    stack_size = (-stack_slot + 7) & ~15

    # Now that we know the total amount of stack space allocated, add a preamble and postamble
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
    for [name, fn] in fns.items():
        insts = allocate_registers(name, fn)
        all_insts = all_insts + insts
    elf_file = elf.create_elf_file(*asm.build(all_insts))
    with open(file, 'wb') as f:
        f.write(bytes(elf_file))
