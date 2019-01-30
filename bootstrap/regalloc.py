import collections

import asm
import elf
import lir

class RegSet:
    def __init__(self, *items):
        if len(items) == 1 and isinstance(items[0], collections.Iterable):
            items = items[0]
        self.items = [i.index if isinstance(i, asm.Register) else i for i in items]
    def add(self, item):
        if isinstance(item, asm.Register):
            item = item.index
        if item not in self.items:
            self.items.insert(0, item)
    def pop(self):
        index = self.items.pop(0)
        return asm.Register(index)
    def copy(self):
        return RegSet(self.items)
    def __getitem__(self, index):
        return asm.Register(self.items[index])
    def __contains__(self, item):
        if isinstance(item, asm.Register):
            item = item.index
        return item in self.items
    def __len__(self):
        return len(self.items)

# All general purpose registers. We can use all registers except RSP and RBP.
ALL_REGISTERS = RegSet(i for i in range(16) if i not in {4, 5})

PARAM_REGS = RegSet(7, 6, 2, 1, 8, 9) # rdi, rsi, rdx, rcx, r8, r9
RETURN_REGS = RegSet(0) # rax
CALLEE_SAVE = RegSet(3, 5, 12, 13, 14, 15) # rbx, rbp, r12, r13, r14, r15
CALLER_SAVE = RegSet(i for i in ALL_REGISTERS if i not in CALLEE_SAVE)

[RSP, RBP] = [asm.Register(4), asm.Register(5)]

def log(*args, **kwargs):
    if 0:
        print(*args, **kwargs)

def get_live_sets(block, outs):
    # Generate the set of live vars at each point in the instruction list
    # We go backwards so we know when things are actually used.
    live_set = outs.copy()
    live_sets = []
    for [i, inst] in reversed(list(enumerate(block.insts))):
        # Add the argument from the last iteration, since we're interested
        # in what's used after the instruction
        live_sets.append(live_set.copy())

        for arg in inst.args:
            if isinstance(arg, lir.Node):
                live_set.add(arg)

        # Make sure we aren't live before we exist
        if inst in live_set:
            live_set.remove(inst)

    # Be sure to put the list back in forward order
    return list(reversed(live_sets))

class VirtualRegister:
    def __init__(self, index):
        self.index = index
    def __str__(self):
        return 'VReg({})'.format(self.index)

def get_live_outs(blocks):
    block_outs = {block: collections.defaultdict(list) for block in blocks}
    phi_reg_assns = {}
    # Use negative numbers for virtual registers so they don't collide with real ones
    phi_id = -2

    for block in blocks:
        for phi in block.phis:
            reg = VirtualRegister(phi_id)
            phi_id -= 1
            phi_reg_assns[phi] = reg
            # Go through all phi operands, which consist of a block and an
            # expression within that block, and make sure we add them to the
            # live outs of the proper block, and mark the assigned register.
            for [src_block, src_op] in zip(block.preds, phi.args):
                block_outs[src_block][src_op].append(phi)

    return [block_outs, phi_reg_assns]

def postorder_traverse(succs, start, used=None):
    if used is None:
        used = set()
    if start not in used:
        used.add(start)
        for succ in succs[start]:
            yield from postorder_traverse(succs, succ, used=used)
        yield start

# Dominance algorithm from http://www.cs.rice.edu/~keith/EMBED/dom.pdf
def get_block_dominance(start, preds, succs):
    # Get postorder traversal minus the first block
    postorder = list(postorder_traverse(succs, start))
    post_id = {b: i for [i, b] in enumerate(postorder)}

    # Function to find the common dominator between two blocks
    def intersect(doms, b1, b2):
        while b1 != b2:
            while post_id[b1] < post_id[b2]:
                b1 = doms[b1]
            while post_id[b1] > post_id[b2]:
                b2 = doms[b2]
        return b1

    doms = {b: (start if b == start else None) for b in preds.keys()}
    changed = True
    while changed:
        changed = False
        for b in reversed(postorder[:-1]):
            rest = preds[b][:]
            new_idom = rest.pop(0)
            assert doms[new_idom] != None
            for p in rest:
                if doms[p] != None:
                    new_idom = intersect(doms, new_idom, p)
            if doms[b] != new_idom:
                doms[b] = new_idom
                changed = True
    return doms

def is_register(reg):
    return (isinstance(reg, asm.Register) or isinstance(reg, VirtualRegister))

def get_arg_reg(arg, reg_assns, clobbered_regs, phi_slot_assns, insts, free_regs,
        force_move=False, assign=False, can_handle_labels=False):
    is_lir_node = isinstance(arg, lir.Node)
    if is_lir_node or force_move:
        orig_arg = arg
        # Anything that isn't a LIR node, we assume is a literal that asm.py can handle
        if is_lir_node:
            if arg in reg_assns:
                arg = reg_assns[arg]
            else:
                arg = phi_slot_assns[arg]
        # Minor optimization: if the instruction can directly handle labels (jmp/call etc),
        # we don't need to lea it
        if isinstance(arg, asm.ExternLabel) and can_handle_labels:
            return arg
        if isinstance(arg, asm.Register) and not force_move:
            return arg
        reg = free_regs.pop()
        if assign:
            reg_assns[orig_arg] = reg
            clobbered_regs.add(reg)
        # Need special handling for labels--use lea of the RIP-relative address, provided by a relocation later
        if isinstance(arg, asm.ExternLabel):
            insts.append(asm.Instruction('lea64', reg, arg))
        else:
            insts.append(asm.Instruction('mov64', reg, arg))
        log('instantiate', insts[-1], free_regs.items)
        return reg
    else:
        return arg

def update_free_regs(node, free_regs, reg_assns, live_set):
    if node not in reg_assns:
        log('not in regs:', node)
        return
    reg = reg_assns[node]
    log('checking', node, reg.to_str(64) if is_register(reg) else reg,
            is_register(reg) and node not in live_set)
    if is_register(reg) and node not in live_set:
        free_regs.add(reg)
        del reg_assns[node]

def allocate_registers(fn):
    [block_outs, phi_reg_assns] = get_live_outs(fn.blocks)

    insts = []

    phi_slot_assns = {}

    clobbered_regs = RegSet()

    for block in fn.blocks:
        for phi in block.phis:
            phi_reg = phi_reg_assns[phi]
            # Stack slot assignment. Phi slots are numbered from -2 to -inf, and
            # stack slots are -8, -16, -24, etc.
            stack_slot = (phi_reg.index + 1) * 8
            phi_slot_assns[phi] = asm.Address(RBP.index, 0, 0, stack_slot)

    for block in fn.blocks:
        insts.append(asm.LocalLabel(block.name))
        log('{}:'.format(insts[-1].name))

        live_sets = get_live_sets(block, set(block_outs[block]))

        for i, [inst, live_set] in enumerate(zip(block.insts, live_sets)):
            log('  i', i, hex(id(inst)), inst)
            for l in live_set:
                log('    ', hex(id(l)), l)

            assert sum(not isinstance(l, lir.Phi) and l.opcode not in {'parameter'}
                    for l in live_set) <= len(ALL_REGISTERS), 'Not enough registers, no spill/fill yet'

        # Now make a run through the instructions. Since we're assuming
        # spill/fill has been taken care of already, this can be done linearly.
        free_regs = ALL_REGISTERS.copy()

        reg_assns = {}

        # Move phis into place. Insert moves from phi slots (that are live ins to this
        # block) to any phi slots where this phi is a live in. Since these are mem-mem
        # copies, we have to use a register intermediate.
        for phi in block.phis:
            for dest in block_outs[block][phi]:
                insts.append(asm.Instruction('mov64', free_regs[0], phi_slot_assns[phi]))
                insts.append(asm.Instruction('mov64', phi_slot_assns[dest], free_regs[0]))
                log('phi-phi', insts[-1], free_regs.items)

        for [inst, live_set] in zip(block.insts, live_sets):
            if inst.opcode in {'parameter', 'literal'}:
                if inst.opcode == 'parameter':
                    src = PARAM_REGS[inst.args[0]]
                elif inst.opcode == 'literal':
                    src = inst.args[0]
                else:
                    assert False
                reg_assns[inst] = src

                # Force a move into the proper slot for literals/parameters that are sources of phis
                if block_outs[block][inst]:
                    # For labels, we need to have an extra lea of the address literal
                    # into a register (provided by a relocation)
                    if isinstance(src, asm.ExternLabel):
                        reg = free_regs[0]
                        insts.append(asm.Instruction('lea64', reg, src))
                        log('label', insts[-1], free_regs.items)
                        src = reg

                    for dest in block_outs[block][inst]:
                        insts.append(asm.Instruction('mov64', phi_slot_assns[dest], src))
                        log('phi literal', insts[-1], free_regs.items)

            elif asm.is_jump_op(inst.opcode):
                insts.append(asm.Instruction(inst.opcode, *inst.args))
            elif inst.opcode == 'call':
                [called_fn, args] = [inst.args[0], inst.args[1:]]
                # Load all arguments from the corresponding stack locations
                assert len(args) <= len(PARAM_REGS)
                call_regs = PARAM_REGS.copy()
                for arg in args:
                    _ = get_arg_reg(arg, reg_assns, clobbered_regs, phi_slot_assns, insts, call_regs,
                            force_move=True)

                called_fn_arg = get_arg_reg(called_fn, reg_assns, clobbered_regs, phi_slot_assns, insts,
                        free_regs, assign=True, can_handle_labels=True)

                insts.append(asm.Instruction(inst.opcode, called_fn_arg))
                log(insts[-1], free_regs.items)

                # Return any now-unused sources to the free set.
                for arg in args:
                    update_free_regs(arg, free_regs, reg_assns, live_set)
                update_free_regs(called_fn, free_regs, reg_assns, live_set)

                # Use the ABI-specified register for pulling out the return value from the call
                reg = RETURN_REGS[0]
            # Regular ops
            else:
                reg = None

                arg_regs = [get_arg_reg(arg, reg_assns, clobbered_regs, phi_slot_assns, insts, free_regs,
                    assign=True) for arg in inst.args]

                # Handle destructive ops first, which might need a move into
                # a new register. Do this before we return any registers to the
                # free set, since the inserted move comes before the instruction.
                destructive = (asm.is_destructive_op(inst.opcode) and
                    arg_regs and is_register(arg_regs[0]))

                if destructive:
                    # See if we can clobber the register. If not, copy it
                    # into another register and change the assignment so
                    # later ops can see it.
                    # XXX This can be done in two ways, but moreover this
                    # should probably be done during scheduling with spill/fill.
                    # This also needs to interact with coalescing, when we have that.
                    if inst.args[0] in live_set:
                        reg = free_regs.pop()
                        insts.append(asm.Instruction('mov64', reg, arg_regs[0]))
                        log('destr copy', insts[-1], free_regs.items)
                        arg_regs[0] = reg
                    else:
                        reg = arg_regs[0]
                        log('clobber', inst.args[0], reg)
                        if inst.args[0] in reg_assns:
                            del reg_assns[inst.args[0]]
                            log('cl free instance')

                # Return any now-unused sources to the free set.
                for arg in inst.args:
                    update_free_regs(arg, free_regs, reg_assns, live_set)

                # Non-destructive ops need a register assignment for their implicit destination
                if not destructive and asm.needs_register(inst.opcode):
                    reg = free_regs.pop()
                    arg_regs = [reg] + arg_regs
                    log('nondestr assign', reg, free_regs.items)

                reg_assns[inst] = reg

                # And now return the destination of the instruction to the
                # free registers, although this only happens when the result
                # is never used...
                if inst not in live_set:
                    log('free inst', inst, arg_regs[0])
                    free_regs.add(arg_regs[0])

                insts.append(asm.Instruction(inst.opcode, *arg_regs))

                log('main', insts[-1], free_regs.items)

                # Copy the result to any phi slots which need it
                for dest in block_outs[block][inst]:
                    insts.append(asm.Instruction('mov64', phi_slot_assns[dest], reg_assns[inst]))
                    log('phi copy', insts[-1])

    # Round up to a multiple of 16
    stack_size = (len(phi_reg_assns)) * 8
    stack_size = (stack_size + 15) & ~15

    save_insts = []
    restore_insts = []
    for reg in clobbered_regs:
        if reg in CALLEE_SAVE:
            save_insts.append(asm.Instruction('push64', reg))
            restore_insts.append(asm.Instruction('pop64', reg))
    # Now that we know the total amount of stack space allocated, add a preamble and postamble
    insts = [asm.GlobalLabel(fn.name),
        asm.Instruction('push64', RBP),
        asm.Instruction('mov64', RBP, RSP),
        asm.Instruction('sub64', RSP, stack_size)
    ] + save_insts + insts + restore_insts + [
        asm.LocalLabel('exit'),
        asm.Instruction('add64', RSP, stack_size),
        asm.Instruction('pop64', RBP),
        asm.Instruction('ret')
    ]

    return insts

# Determine all predecessor and successor blocks
def get_block_linkage(blocks):
    preds = {block.name: [] for block in blocks}
    succs = {}

    for [i, block] in enumerate(blocks):
        last_inst = block.insts[-1]
        [opcode, args] = [last_inst[0], last_inst[1:]]
        dests = []
        # Jump: add the destination block and the fall-through block
        if asm.is_jump_op(opcode):
            [dest] = args
            # XXX Look for labels
            if isinstance(dest, asm.Label):
                dest = dest.name
            dests = [dest]
            if opcode != 'jmp':
                assert i + 1 < len(blocks)
                dests = dests + [i + 1]
        # ...or just the fallthrough
        elif i + 1 < len(blocks):
            dests = [i + 1]

        # Link up blocks
        succs[i] = dests
        for dest in dests:
            # XXX why are these different types?
            if isinstance(dest, int):
                dest = blocks[dest].name
            preds[dest].append(i)

    return [preds, succs]

def print_insts(insts):
    for inst in insts:
        if isinstance(inst, asm.Label):
            print('{}:'.format(inst.name))
        else:
            print('    {}'.format(inst))

def export_functions(file, fns):
    all_insts = []
    for fn in fns:
        insts = allocate_registers(fn)
        all_insts = all_insts + insts
        print_insts(insts)

    elf_file = elf.create_elf_file(*asm.build(all_insts))
    with open(file, 'wb') as f:
        f.write(bytes(elf_file))
