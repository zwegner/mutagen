import regalloc
import elf
import lir
# XXX work around weird importing behavior
asm = regalloc.asm

class Function(parameters, blocks, node_map):
    pass

class BasicBlock(name, phis, insts):
    pass

PARAM_REGS = [7, 6, 2, 1]
RETURN_REGS = [0]

def get_arg_reg(arg, reg_assns, free_regs, insts, force_move=False):
    if isinstance(arg, lir.NR) or force_move:
        if isinstance(arg, lir.NR):
            arg = reg_assns[arg.node_id]
        [reg, free_regs] = [free_regs[0], free_regs[1:]]
        reg = asm.Register(reg)
        insts = insts + [asm.Instruction('mov64', reg, arg)]
        return [reg, free_regs, insts]
    else:
        return [arg, free_regs, insts]

def allocate_registers(name, fn):
    stack_slot = -8
    insts = []

    [rsp, rbp, rax] = [asm.Register(4), asm.Register(5), asm.Register(0)]
    all_registers = [r for r in (set(range(16)) - set(PARAM_REGS) - {rsp.index, rbp.index})]

    reg_assns = phi_reg_assns = {}

    # XXX meh
    inv_node_map = {}
    for [node_id, [block_name, index]] in fn.node_map:
        if block_name not in inv_node_map:
            inv_node_map = inv_node_map <- [block_name] = {}
        inv_node_map = inv_node_map <- [block_name][index] = node_id

    # Assign phi stack slots. Since multiple phis can read the same value, we
    # need to maintain a list of slot assignments, not just one. Each value that
    # is used by phis will later store its value in each of those registers.
    for block in fn.blocks:
        for phi in block.phis:
            node_id = inv_node_map[block.name][phi.args[0]]

            # Assign a slot
            dest = asm.Address(rbp.index, 0, 0, stack_slot)
            stack_slot = stack_slot - 8
            assert node_id not in reg_assns
            reg_assns = reg_assns <- [node_id] = dest

            # Add it to the phi list of each phi argument
            for src_node in phi.args:
                if src_node not in phi_reg_assns:
                    phi_reg_assns = phi_reg_assns <- [src_node] = []
                phi_reg_assns = phi_reg_assns <- [src_node] += [dest]

    for block in fn.blocks:
        insts = insts + [asm.LocalLabel(block.name)]

        for phi in block.phis:
            assert phi.opcode == 'phi'
            node_id = inv_node_map[block.name][phi.args[0]]
            # We assigned stack slots for phis above, but we delegate the responsibility
            # of writing to the slot to the source instructions in all predecessor blocks.
            # Given that, we need to check if this phi is a source for another phi.
            if node_id in phi_reg_assns:
                reg = asm.Register(all_registers[0])
                # For each phi that reads this phi value, we need two movs, since
                # x86 doesn't have mem->mem copies.
                for phi in phi_reg_assns[node_id]:
                    insts = insts + [asm.Instruction('mov64', reg,
                            reg_assns[node_id]),
                        asm.Instruction('mov64', phi, reg)]

        for [inst_id, inst] in enumerate(block.insts):
            [opcode, args] = [inst.opcode, inst.args]

            node_id = None
            if block.name in inv_node_map and inst_id in inv_node_map[block.name]:
                node_id = inv_node_map[block.name][inst_id]
            else:
                assert opcode in ['test64', 'jnz', 'jmp']

            # Return: optionally move the argument into eax and jump to the exit block.
            # We jump since we need to take care of the stack but we don't yet know how
            # much space we allocate.
            if opcode == 'ret':
                if args:
                    assert len(args) == 1
                    # Put the return value in a register. Since the return value isn't actually
                    # part of the ret instruction, be sure to force a move to rax in case the
                    # return value is a literal.
                    [arg, free_regs, insts] = get_arg_reg(args[0], reg_assns, RETURN_REGS, insts, force_move=True)
                insts = insts + [asm.Instruction('jmp', asm.LocalLabel('exit'))]
            # Jump instructions
            elif asm.is_jump_op(opcode):
                # Make sure only the last instruction is a control flow op
                # XXX not true at the moment, we have two jumps at the end of every branching block
                #assert inst_id == len(block.insts) - 1

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
                    reg = asm.Register(PARAM_REGS[arg])
                elif opcode == 'call':
                    [called_fn, args] = [args[0], args[1:]]
                    # Load all arguments from the corresponding stack locations
                    assert len(args) <= len(PARAM_REGS)
                    free_regs = PARAM_REGS
                    for arg in args:
                        [_, free_regs, insts] = get_arg_reg(arg, reg_assns, free_regs, insts, force_move=True)

                    insts = insts + [asm.Instruction(opcode, called_fn)]
                    # C ABI dictates the return value from function is in rax
                    reg = rax
                elif args:
                    # Load all arguments from the corresponding stack locations. Anything
                    # that isn't an NR, we assume is a literal
                    free_regs = all_registers
                    arg_regs = []
                    for arg in args:
                        [arg, free_regs, insts] = get_arg_reg(arg, reg_assns, free_regs, insts)
                        arg_regs = arg_regs + [arg]

                    # Assign a destination to 3-operand instructions
                    if asm.needs_register(opcode) and not (asm.is_destructive_op(opcode) and
                            isinstance(arg_regs[0], asm.Register)):
                        [reg, free_regs] = [free_regs[0], free_regs[1:]]
                        arg_regs = [asm.Register(reg)] + arg_regs
                    reg = arg_regs[0]

                    # Add the instruction as well as a store into our stack slot
                    insts = insts + [asm.Instruction(opcode, *arg_regs)]

                # Store the result on the stack
                if asm.needs_register(opcode):
                    # Assign a stack slot
                    dest = asm.Address(rbp.index, 0, 0, stack_slot)
                    stack_slot = stack_slot - 8
                    reg_assns = reg_assns <- [node_id] = dest

                    insts = insts + [asm.Instruction('mov64', dest, reg)]
                    # ...and again for any phi that references it
                    if node_id in phi_reg_assns:
                        for phi in phi_reg_assns[node_id]:
                            insts = insts + [asm.Instruction('mov64', phi, reg)]
                else:
                    assert node_id not in phi_reg_assns

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
    for [name, fn] in fns:
        insts = allocate_registers(name, fn)
        print_insts(insts)
        all_insts = all_insts + insts
    elf_file = elf.create_elf_file(*asm.build(all_insts))
    write_binary_file(file, elf_file)
