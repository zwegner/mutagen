import collections

import asm
import elf
import lir

class RegSet:
    def __init__(self, *items):
        if len(items) == 1 and isinstance(items[0], collections.Iterable):
            items = items[0]
        self.items = [i.index if isinstance(i, asm.GPReg) else i for i in items]
    def add(self, item):
        if isinstance(item, asm.GPReg):
            item = item.index
        if item not in self.items:
            self.items.insert(0, item)
    def pop(self):
        index = self.items.pop(0)
        return asm.GPReg(index)
    def copy(self):
        return RegSet(self.items)
    def __getitem__(self, index):
        return asm.GPReg(self.items[index])
    def __contains__(self, item):
        if isinstance(item, asm.GPReg):
            item = item.index
        return item in self.items
    def __len__(self):
        return len(self.items)
    def __str__(self):
        return 'RegSet(%s)' % ', '.join(str(self[i]) for i in range(len(self)))

# All general purpose registers. We can use all registers except RSP and RBP.
ALL_REGISTERS = RegSet(i for i in range(16) if i not in {4, 5})

PARAM_REGS = RegSet(7, 6, 2, 1, 8, 9) # rdi, rsi, rdx, rcx, r8, r9
RETURN_REGS = RegSet(0) # rax
CALLEE_SAVE = RegSet(3, 5, 12, 13, 14, 15) # rbx, rbp, r12, r13, r14, r15
CALLER_SAVE = RegSet(i for i in ALL_REGISTERS if i not in CALLEE_SAVE)

FREE_REGS = RegSet(sorted(ALL_REGISTERS, key=lambda r: r in CALLER_SAVE))

[RSP, RBP] = [asm.GPReg(4), asm.GPReg(5)]

def log(*args, **kwargs):
    if 0:
        print(*args, **kwargs)

def get_live_sets(block, last_uses):
    live_set = set(block.phi_selects.values()) | {block.phi_write}
    for inst in block.insts:
        for arg in inst.args:
            if last_uses.get(arg) is inst:
                live_set.remove(arg)
                last_uses.pop(arg)
        if inst in last_uses:
            live_set.add(inst)
        yield live_set.copy()

def get_last_uses(block, outs):
    last_uses = {v: None for [k, v] in outs.items()}
    for inst in reversed(block.insts):
        for arg in inst.args:
            if isinstance(arg, lir.Node) and arg not in last_uses:
                last_uses[arg] = inst

    return last_uses

class VirtualRegister:
    def __init__(self, index):
        self.index = index
    def __str__(self):
        return 'VReg({})'.format(self.index)

def block_label(block):
    return asm.LocalLabel(block.name)

# Fix up the CFG to split critical edges. Critical edges are edges in the CFG
# that link a block with multiple successors to one with multiple predecessors.
# Our register allocator can't deal with these, because phis (that shuffle
# variables around) semantically happen on the control flow edges, and thus the
# instructions to handle them have to be unique for a (pred, succ) pair. For
# non-critical edges, the instructions can be inserted at the end of the
# predecessor or the beginning of the successor, but for critical edges there's
# no place for them. So we insert a new basic block with no instructions except
# what will later be added to implement the phi, and a jump at the end.
def split_critical_edges(fn):
    new_blocks = []
    for block in fn.blocks:
        new_blocks.append(block)
        if len(block.succs) > 1:
            for si, succ in enumerate(block.succs):
                if len(succ.preds) > 1:
                    for i, pred in enumerate(succ.preds):
                        if pred is block:
                            pi = i
                            break
                    else:
                        assert False

                    # Create new phis, and fix up the successor's phis to point
                    # to them
                    phis = [lir.Phi(p.name, [p.args[pi]]) for p in succ.phis]
                    for p, phi in zip(succ.phis, phis):
                        p.args[pi] = phi

                    name = 'block$split$%s$%s' % (block.name, succ.name)
                    split_block = lir.BasicBlock(name, phis, [], None,
                            [block], [succ])

                    new_blocks.append(split_block)
                    block.succs[si] = split_block
                    succ.preds[pi] = split_block

    fn.blocks = new_blocks

# Create a physical ordering: for now just use the order they were created in
def finalize_cfg(fn):
    phys_idx = 0
    for block in fn.blocks:
        block.phys_idx = phys_idx
        phys_idx += 1
    return phys_idx

# Get the jump instructions to the proper successors at the end of a block,
# based on the physical block layout
def get_jumps(block, exit_phys_idx, exit_label):
    next_phys = block.phys_idx + 1
    # Add a conditional jump if needed
    if block.test:
        assert len(block.succs) == 2
        assert block.test is block.insts[-1]
        if block.succs[0].phys_idx == next_phys:
            yield lir.jz(block_label(block.succs[1]))
        elif block.succs[1].phys_idx == next_phys:
            yield lir.jnz(block_label(block.succs[0]))
        else:
            # If neither successor is the next physical block, we need two jumps
            yield lir.jnz(block_label(block.succs[0]))
            yield lir.jmp(block_label(block.succs[1]))
    elif len(block.succs) == 1:
        # Add a jump if we're not going to the next physical block
        if block.succs[0].phys_idx != next_phys:
            yield lir.jmp(block_label(block.succs[0]))
    else:
        assert not block.succs
        # Add a jump to the exit block, if the exit block isn't the last
        if next_phys != exit_phys_idx:
            yield lir.jmp(exit_label)

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
    return (isinstance(reg, asm.GPReg) or isinstance(reg, VirtualRegister))

def instantiate_reg(node, free_regs, clobbered_regs, insts):
    reg = free_regs.pop()
    clobbered_regs.add(reg)
    # Need special handling for labels--use lea of the RIP-relative
    # address, provided by a relocation later
    if isinstance(node, asm.ExternLabel):
        insts.append(asm.Instruction('lea', reg, node))
    else:
        insts.append(asm.Instruction('mov', reg, node))
    return reg

def get_arg_reg(arg, reg_assns, stack_assns, free_regs, clobbered_regs, insts,
        force_move=False, assign=False, can_handle_labels=False):
    is_lir_node = isinstance(arg, lir.Node)
    if is_lir_node or force_move:
        orig_arg = arg
        # Anything that isn't a LIR node, we assume is a literal that asm.py
        # can handle
        if is_lir_node:
            if arg in reg_assns:
                arg = reg_assns[arg]
            else:
                arg = stack_assns[arg]
        # Minor optimization: if the instruction can directly handle labels
        # (jmp/call etc), we don't need to lea it
        if isinstance(arg, asm.ExternLabel) and can_handle_labels:
            return arg
        if isinstance(arg, asm.GPReg) and not force_move:
            return arg

        reg = instantiate_reg(arg, free_regs, clobbered_regs, insts)
        if assign:
            reg_assns[orig_arg] = reg

        log('instantiate', insts[-1], free_regs)
        return reg
    else:
        return arg

def update_free_regs(node, free_regs, reg_assns, live_set):
    if node not in reg_assns:
        log('not in regs:', node)
        return
    reg = reg_assns[node]
    log('checking', hex(id(node)), node, reg,
            is_register(reg) and node not in live_set)
    log('   live', live_set)
    if is_register(reg) and node not in live_set:
        free_regs.add(reg)
        del reg_assns[node]
    log('   free', free_regs)

# Algorithm from Section 4.4, Implementing Φ-operations, "Register Allocation
# for Programs in SSA Form", Sebastian Hack 2006.
def parallel_copy(reads, writes, free_regs):
    assert len(reads) == len(writes)
    edges = collections.defaultdict(list)
    for [r, w] in zip(reads, writes):
        edges[r].append(w)

    insts = []

    # First, do moves where the destination isn't read. We forward other reads
    # of the source to the destination (except src->src self loops) so we can
    # free registers quickly
    done = False
    while not done:
        done = True
        for [r, ws] in edges.items():
            for [i, w] in enumerate(ws):
                if len(edges[w]) == 0:
                    insts.append(('mov', w, r))
                    edges[w].extend(w2 for w2 in ws if w2 != r)
                    del edges[r]
                    done = False
                    free_regs.add(r)
                    break

    # Implement cycles in the register transfer graph. If we have free
    # registers, we can use a temporary, otherwise we have to use a bunch
    # of swaps, or allocate a stack slot. This should be configurable, but
    # for now just use swaps since x86 has the cheap-ish xchg instruction.
    while edges:
        r = min(edges, key=lambda r: r.index)
        [w] = edges.pop(r)
        if r != w:
            cycle = [r]
            first = r
            while w != first:
                cycle.append(w)
                [w] = edges.pop(w)

            if free_regs:
                temp = free_regs.pop()
                last = cycle[-1]
                insts.append(('mov', temp, last))
                for reg in reversed(cycle[:-1]):
                    insts.append(('mov', last, reg))
                    last = reg
                insts.append(('mov', last, temp))

            else:
                last = cycle[0]
                for r in cycle[1:]:
                    insts.append(('swap', r, last))

    return insts

def move_phi_args(phi_read, phi_write, free_regs):
    keys = list(sorted(phi_write.keys()))
    return parallel_copy([phi_read[k] for k in keys],
            [phi_write[k] for k in keys], free_regs)

def gen_save_insts(regs, extra_stack=0):
    [save_insts, restore_insts] = [[asm.Instruction(op, reg)
            for reg in regs] for op in ['push', 'pop']]
    restore_insts = restore_insts[::-1]

    # Round up to a multiple of 16
    stack_size = len(save_insts) * 8
    stack_adj = ((stack_size + extra_stack + 15) & ~15) - stack_size

    if not stack_adj:
        return [save_insts, restore_insts]

    # Now that we know the total amount of stack space allocated, add a
    # prologue and epilogue
    save_insts = [asm.Instruction('sub', RSP, stack_adj), *save_insts]
    restore_insts = [*restore_insts, asm.Instruction('add', RSP, stack_adj)]

    return [save_insts, restore_insts]

def allocate_registers(fn):
    split_critical_edges(fn)

    for block in fn.blocks:
        log('{}:'.format(block.name))

        log('  w', hex(id(block.phi_write)), block.phi_write)

        for i, inst in enumerate(block.insts):
            log('  i', i, hex(id(inst)), inst)

        log('  r', hex(id(block.phi_read)), block.phi_read)

    # Stack slots for spill/fill etc. Not used right now
    stack_assns = {}

    # Registers assigned to all arguments to each phi read or write
    phi_assns = {}

    # All registers touched by the function, that need to be saved/restored
    # XXX should track this in an easier/more robust way
    clobbered_regs = RegSet()

    # A few dicts for keeping track of per-basic-block info that needs to
    # be accessed elsewhere
    block_free_regs = {b: FREE_REGS.copy() for b in fn.blocks}
    block_reg_assns = {b: {} for b in fn.blocks} # ok, only accessed for logging...
    block_insts = {b: [] for b in fn.blocks}

    for block in fn.blocks:
        log('{}:'.format(block.name))

        for i, inst in enumerate(block.insts):
            log('  i', i, hex(id(inst)), inst)

    for block in fn.blocks:
        insts = block_insts[block]

        last_uses = get_last_uses(block, block.phi_read.args)
        live_set_iter = list(get_live_sets(block, last_uses))

        log('\n{}:'.format(block.name))
        log('  ins:', block.phi_selects)
        log('  outs:', block.phi_read.args)
        log('  last:', last_uses)

        for i, [inst, live_set] in enumerate(zip(block.insts, live_set_iter)):
            pressure = sum(not isinstance(l, lir.PhiSelect) and
                                l.opcode not in {'parameter'} for l in live_set)
            assert pressure <= len(FREE_REGS), 'Not enough registers'

        free_regs = block_free_regs[block]
        reg_assns = block_reg_assns[block]

        # Allocate registers for the phi write. We'll make sure elsewhere that
        # the arguments are moved to the right registers
        if block.succs:
            regs = {name: free_regs.pop() for name in sorted(block.phi_write.args)}
        # For the exit block, we should only have at most one live value, the
        # return value (which is stored in a special variable). Assign it to
        # the ABI's return register.
        else:
            regs = {}
            if block.phi_write.args:
                assert list(block.phi_write.args.keys()) == ['$return_value']
                regs['$return_value'] = RETURN_REGS[0]
        phi_assns[block.phi_write] = regs

        # Now make a run through the instructions. Since we're assuming
        # spill/fill has been taken care of already, this can be done linearly.
        for [inst, live_set] in zip(block.insts, live_set_iter):
            if isinstance(inst, lir.PhiSelect):
                reg = phi_assns[inst.phi_write][inst.name]
                reg_assns[inst] = reg
                log('phi assign', hex(id(inst)), inst, reg, free_regs)

            # Returns are sorta pseudo-ops that get eliminated. Basically
            # they're only there to the SSA machinery sees the $return_value
            # special variable as a live in to the exit block.
            elif inst.opcode == 'return':
                pass

            # Instantiate parameters/literals into registers. This is usually
            # a waste but right now needed for correctness
            elif inst.opcode == 'parameter':
                [index] = inst.args
                src = PARAM_REGS[index]
                reg = instantiate_reg(src, free_regs, clobbered_regs, insts)
                reg_assns[inst] = reg
                log('param', inst, src, block.phi_read.args.get(inst))

            elif inst.opcode == 'literal':
                [src] = inst.args
                reg = instantiate_reg(src, free_regs, clobbered_regs, insts)
                reg_assns[inst] = reg
                log('lit', inst, src, block.phi_read.args.get(inst))

            elif inst.opcode == 'call':
                [called_fn, args] = [inst.args[0], inst.args[1:]]
                # Load all arguments from the corresponding stack locations
                assert len(args) <= len(PARAM_REGS)
                call_regs = PARAM_REGS.copy()
                for arg in args:
                    _ = get_arg_reg(arg, reg_assns, stack_assns, call_regs,
                            clobbered_regs, insts, force_move=True)

                called_fn_arg = get_arg_reg(called_fn, reg_assns, stack_assns,
                        free_regs, clobbered_regs, insts, can_handle_labels=True)

                [save_insts, restore_insts] = gen_save_insts(
                        [reg_assns[n] for n in live_set if n in reg_assns and
                            reg_assns[n] in CALLER_SAVE])

                call = asm.Instruction(inst.opcode, called_fn_arg)

                insts += [*save_insts, call, *restore_insts]

                log(call, free_regs)

                # Return any now-unused sources to the free set.
                for arg in args:
                    update_free_regs(arg, free_regs, reg_assns, live_set)
                update_free_regs(called_fn, free_regs, reg_assns, live_set)

                # Use the ABI-specified register for pulling out the return
                # value from the call. Move it to a fresh register so we can
                # keep it alive
                reg = instantiate_reg(RETURN_REGS[0], free_regs, clobbered_regs, insts)
                reg_assns[inst] = reg

            # Regular ops
            else:
                reg = None

                arg_regs = [get_arg_reg(arg, reg_assns, stack_assns, free_regs,
                        clobbered_regs, insts, assign=True) for arg in inst.args]

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
                        clobbered_regs.add(reg)
                        insts.append(asm.Instruction('mov', reg, arg_regs[0]))
                        log('destr copy', insts[-1], free_regs)
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

                # Non-destructive ops need a register assignment for their
                # implicit destination
                if not destructive and asm.needs_register(inst.opcode):
                    reg = free_regs.pop()
                    clobbered_regs.add(reg)
                    arg_regs = [reg] + arg_regs
                    log('nondestr assign', reg, free_regs)

                reg_assns[inst] = reg

                # And now return the destination of the instruction to the
                # free registers, although this only happens when the result
                # is never used...
                if inst not in live_set:
                    log('free inst', inst, arg_regs[0])
                    free_regs.add(arg_regs[0])

                insts.append(asm.Instruction(inst.opcode, *arg_regs))

                log('main', insts[-1], free_regs)

        # Remember allocated registers for the phi read
        regs = {name: reg_assns[inst] for [name, inst] in
                sorted(block.phi_read.args.items())}
        phi_assns[block.phi_read] = regs

    for block in fn.blocks:
        log('assns for', block.name, block_reg_assns[block])

    # Undefined variable check, should have a real message
    assert not fn.blocks[0].phi_write.args, fn.blocks[0].phi_write.args

    # Implement phis. For each CFG edge, we do a parallel move from the phi read
    # phase to the phi write phase, moving all registers that are needed in the
    # successor block (not all successors use all arguments of the read). The
    # actual move/swap instructions for the parallel move must go in either the
    # predecessor or successor, depending on which is unique (see
    # split_critical_edges() for an explanation how/why we ensure this is true)
    for block in fn.blocks:
        phi_r = phi_assns[block.phi_read]
        first = len(block.succs) == 1
        for succ in block.succs:
            phi_w = phi_assns[succ.phi_write]
            free_regs = block_free_regs[block] if first else block_free_regs[succ]
            insts = []
            for [inst, dst, src] in move_phi_args(phi_r, phi_w, free_regs):
                if inst == 'mov':
                    insts.append(asm.Instruction('mov', dst, src))
                else:
                    assert inst == 'swap'
                    insts.append(asm.Instruction('xchg', src, dst))

            if first:
                block_insts[block].extend(insts)
            else:
                block_insts[succ][:0] = insts

    # Kind of a hack: create an exit label just for this function. Labels need
    # to be unique in the ELF file we generate, so all the jumps get patched
    # properly. OK, so that's cool. We only need to do this for the exit block
    # because every other block gets a globally unique name like block$7
    exit_label = asm.LocalLabel('%s$exit' % fn.name)

    # Choose a physical layout of basic blocks, and collect all instructions
    # from each basic block with any jumps needed for the chosen layout
    exit_phys_idx = finalize_cfg(fn)
    insts = []
    for block in fn.blocks:
        insts.append(asm.LocalLabel(block.name))
        insts.extend(block_insts[block])

        # Add jump instructions
        for jmp in get_jumps(block, exit_phys_idx, exit_label):
            assert asm.is_jump_op(jmp.opcode)
            insts.append(asm.Instruction(jmp.opcode, *jmp.args))

    # Generate push/pop instructions for callee-save instructions, and
    # adjust the stack for phi
    [save_insts, restore_insts] = gen_save_insts(
            [reg for reg in clobbered_regs if reg in CALLEE_SAVE],
            extra_stack=len(stack_assns)*8)

    # Now that we know the total amount of stack space allocated, add a
    # prologue and epilogue
    insts = [
        asm.GlobalLabel(fn.name),
        asm.Instruction('push', RBP),
        asm.Instruction('mov', RBP, RSP),
        *save_insts,
        *insts,
        exit_label,
        *restore_insts,
        asm.Instruction('pop', RBP),
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
    for fn in fns:
        insts = allocate_registers(fn)
        all_insts = all_insts + insts
        print_insts(insts)

    elf_file = elf.create_elf_file(*asm.build(all_insts))
    with open(file, 'wb') as f:
        f.write(bytes(elf_file))
