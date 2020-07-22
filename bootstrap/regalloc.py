import collections

import asm
import elf
import lir

class RegSet:
    def __init__(self, reg_type, *items):
        if len(items) == 1 and isinstance(items[0], collections.Iterable):
            items = items[0]
        self.reg_type = reg_type
        self.items = []
        for i in items:
            if not isinstance(i, int):
                assert isinstance(i, reg_type)
                i = i.index
            self.items.append(i)
    def add(self, item):
        if not isinstance(item, int):
            assert isinstance(item, self.reg_type)
            item = item.index
        if item not in self.items:
            self.items.insert(0, item)
    def pop(self):
        index = self.items.pop(0)
        return self.reg_type(index)
    def copy(self):
        return RegSet(self.reg_type, self.items)
    def __getitem__(self, index):
        return self.reg_type(self.items[index])
    def __contains__(self, item):
        if isinstance(item, self.reg_type):
            return item.index in self.items
        return False
    def __len__(self):
        return len(self.items)
    def __repr__(self):
        return 'RegSet(%s)' % ', '.join(str(self[i]) for i in range(len(self)))

# All general purpose registers. We can use all registers except RSP and RBP.
ALL_REGISTERS = RegSet(asm.GPReg, (i for i in range(16) if i not in {4, 5}))

PARAM_REGS = RegSet(asm.GPReg, 7, 6, 2, 1, 8, 9) # rdi, rsi, rdx, rcx, r8, r9
RETURN_REGS = RegSet(asm.GPReg, 0) # rax
CALLEE_SAVE = RegSet(asm.GPReg, 3, 5, 12, 13, 14, 15) # rbx, rbp, r12, r13, r14, r15
CALLER_SAVE = RegSet(asm.GPReg, (i for i in ALL_REGISTERS if i not in CALLEE_SAVE))
FREE_REGS = RegSet(asm.GPReg, sorted(ALL_REGISTERS, key=lambda r: r in CALLER_SAVE))

VEC_FREE_REGS = RegSet(asm.VecReg, list(range(16)))
VEC_CALLER_SAVE = VEC_FREE_REGS

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
    return isinstance(reg, asm.Register)

REG_TYPES = [asm.GPReg, asm.VecReg]

# HACK
def get_result_type(node: lir.Node):
    if isinstance(node, asm.Register):
        return type(node)
    # XXX
    if isinstance(node, (asm.Address, asm.Label)):
        return asm.GPReg
    if isinstance(node, lir.Inst):
        for table in [asm.avx_table, asm.sse_table]:
            if node.opcode in table:
                [*_, forms, _] = table[node.opcode]
                # Return the type of the first argument of the first form
                return forms[0][0][0]
        return asm.GPReg
    assert False, str(node)

class RegAllocContext:
    def __init__(self, fn):
        self.fn = fn
        self.block_insts = {b: [] for b in fn.blocks}
        self.free_regs = None
        self.free_regs_in = {b: {} for b in fn.blocks}
        self.free_regs_out = {b: {} for b in fn.blocks}
        self.block_reg_assns = {b: {} for b in fn.blocks}
        self.current_block = None
        # All registers touched by the function, that need to be saved/restored
        self.clobbered_regs = {t: RegSet(t) for t in REG_TYPES}
        # Stack slots for spill/fill etc. Not used right now
        self.stack_assns = {}

    def start_block(self, block, free_regs):
        self.current_block = block
        self.insts = self.block_insts[block]
        self.free_regs = free_regs
        for t in REG_TYPES:
            self.free_regs_in[block][t] = free_regs[t].copy()
        self.reg_assns = self.block_reg_assns[block]

    def end_block(self, block):
        for t in REG_TYPES:
            self.free_regs_out[block][t] = self.free_regs[t].copy()
        self.current_block = None
        self.reg_assns = None
        self.free_regs = None
        self.insts = None

    def alloc_reg(self, reg_type):
        reg = self.free_regs[reg_type].pop()
        self.clobbered_regs[reg_type].add(reg)
        return reg

    def dealloc_reg(self, reg, node=None):
        reg_type = type(reg)
        self.free_regs[reg_type].add(reg)
        if node:
            del self.reg_assns[node]

    def instantiate_reg(self, node, free_regs=None):
        if isinstance(node, asm.Data):
            return node
        reg_type = get_result_type(node)
        if free_regs is None:
            free_regs = self.free_regs[reg_type]
        assert free_regs.reg_type is reg_type, (free_regs.reg_type, reg_type)
        reg = free_regs.pop()
        self.clobbered_regs[reg_type].add(reg)
        # Need special handling for labels--use lea of the RIP-relative
        # address, provided by a relocation later
        if isinstance(node, asm.ExternLabel):
            self.insts.append(asm.Instruction('lea', reg, node))
        elif reg_type is asm.GPReg:
            self.insts.append(asm.Instruction('mov', reg, node))
        else:
            # XXX which vector size to use?
            self.insts.append(asm.Instruction('vmovdqu', reg, node))
        log('instantiate', self.insts[-1], free_regs)
        return reg

    def get_arg_reg(self, arg, free_regs=None, force_move=False, assign=False,
            can_handle_labels=False):
        is_lir_node = isinstance(arg, lir.Node)
        if is_lir_node or force_move:
            orig_arg = arg
            # Anything that isn't a LIR node, we assume is a literal that asm.py
            # can handle
            if is_lir_node:
                if arg in self.reg_assns:
                    arg = self.reg_assns[arg]
                elif isinstance(arg, lir.Inst) and arg.opcode == 'literal':
                    [arg] = arg.args
                else:
                    arg = self.stack_assns[arg]
            # Minor optimization: if the instruction can directly handle labels
            # (jmp/call etc), we don't need to lea it
            if isinstance(arg, asm.ExternLabel) and can_handle_labels:
                return arg
            if isinstance(arg, asm.GPReg) and not force_move:
                return arg

            reg = self.instantiate_reg(arg, free_regs=free_regs)
            if assign:
                self.reg_assns[orig_arg] = reg

            return reg
        else:
            return arg

    def update_free_regs(self, node, live_set):
        if node not in self.reg_assns:
            log('not in regs:', node)
            return
        reg = self.reg_assns[node]
        log('checking', hex(id(node)), node, reg,
                is_register(reg) and node not in live_set)
        if is_register(reg) and node not in live_set:
            self.dealloc_reg(reg, node=node)

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

def move_phi_args(phi_read, phi_write, keys, free_regs):
    keys = list(sorted(keys))
    return parallel_copy([phi_read[k] for k in keys],
            [phi_write[k] for k in keys], free_regs)

def gen_save_insts(live_regs, extra_stack=0):
    regs = {asm.GPReg: [], asm.VecReg: []}
    for reg in live_regs:
        regs[get_result_type(reg)].append(reg)

    [save_insts, restore_insts] = [[asm.Instruction(op, reg)
            for reg in regs[asm.GPReg]] for op in ['push', 'pop']]
    restore_insts = restore_insts[::-1]

    stack_size = len(save_insts) * 8

    for [i, reg] in enumerate(regs[asm.VecReg]):
        addr = asm.Address(RSP.index, 0, 0, 32 * i, size=256)
        save_insts.append(asm.Instruction('vmovdqu', addr, reg))
        restore_insts.insert(0, asm.Instruction('vmovdqu', reg, addr))

    # XXX 32 bytes
    extra_stack += 32 * len(regs[asm.VecReg])

    # Round up to a multiple of 16
    stack_adj = ((stack_size + extra_stack + 15) & ~15) - stack_size

    if stack_adj:
        save_insts = [asm.Instruction('sub', RSP, stack_adj)] + save_insts
        restore_insts += [asm.Instruction('add', RSP, stack_adj)]

    return [save_insts, restore_insts]

def allocate_registers(fn):
    split_critical_edges(fn)

    for block in fn.blocks:
        log('{}:'.format(block.name))

        log('  w', hex(id(block.phi_write)), block.phi_write)

        for i, inst in enumerate(block.insts):
            log('  i', i, hex(id(inst)), inst)

        log('  r', hex(id(block.phi_read)), block.phi_read)

    # Registers assigned to all arguments to each phi read or write
    phi_assns = {}

    ctx = RegAllocContext(fn)

    for block in fn.blocks:
        log('{}:'.format(block.name))

        for i, inst in enumerate(block.insts):
            log('  i', i, hex(id(inst)), inst)

    # Analyze phi types. We need to have the types up front so we can allocate the
    # registers for the writes at the start of each block
    phi_types = {}
    for block in fn.blocks:
        phi_types[block] = {}
        for name in block.phi_write.args:
            types = set()
            for pred in block.preds:
                arg = pred.phi_read.args[name]
                if not isinstance(arg, lir.PhiSelect):
                    types.add(get_result_type(arg))
                # Use a one-block lookbehind to see if we can determine the type
                # XXX need a proper iterative solution
                elif pred in phi_types and name in phi_types[pred]:
                    types.add(phi_types[pred][name])
            assert len(types) == 1, ('could not determine type of %s '
                    'for block %s: %s' % (name, block.name, types))
            [phi_t] = types
            assert phi_t in REG_TYPES
            phi_types[block][name] = phi_t

    # The main register allocation loop
    for block in fn.blocks:
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

        free_regs = {asm.GPReg: FREE_REGS.copy(), asm.VecReg: VEC_FREE_REGS.copy()}
        reg_assns = ctx.block_reg_assns[block]

        # Allocate registers for the phi write. We'll make sure elsewhere that
        # the arguments are moved to the right registers
        if block.succs:
            regs = {name: free_regs[phi_types[block][name]].pop()
                    for [name, arg] in sorted(block.phi_write.args.items())}
        # For the exit block, we should only have at most one live value, the
        # return value (which is stored in a special variable). Assign it to
        # the ABI's return register.
        else:
            regs = {}
            if block.phi_write.args:
                assert list(block.phi_write.args.keys()) == ['$return_value']
                regs['$return_value'] = RETURN_REGS[0]
        phi_assns[block.phi_write] = regs

        ctx.start_block(block, free_regs)

        # Now make a run through the instructions. Since we're assuming
        # spill/fill has been taken care of already, this can be done linearly.
        for [inst, live_set] in zip(block.insts, live_set_iter):
            log('handling', inst)
            log('   free gpr', ctx.free_regs[asm.GPReg])
            log('   free vec', ctx.free_regs[asm.VecReg])
            log('   live', live_set)

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
                reg = ctx.instantiate_reg(src)
                reg_assns[inst] = reg
                log('param', inst, src, block.phi_read.args.get(inst))

            elif inst.opcode == 'literal':
                [value] = inst.args
                if not isinstance(value, int):
                    reg = ctx.instantiate_reg(value)
                    reg_assns[inst] = reg
                    log('lit', inst, value, block.phi_read.args.get(inst))

            elif inst.opcode == 'call':
                [called_fn, args] = [inst.args[0], inst.args[1:]]
                # Load all arguments from the corresponding stack locations
                assert len(args) <= len(PARAM_REGS)
                call_regs = PARAM_REGS.copy()
                for arg in args:
                    _ = ctx.get_arg_reg(arg, free_regs=call_regs, force_move=True)

                called_fn_arg = ctx.get_arg_reg(called_fn, can_handle_labels=True)

                # Generate save/restore instructions for caller-save registers
                clobbered = [reg for reg in reg_assns.values()
                        if reg in VEC_CALLER_SAVE or reg in CALLER_SAVE]
                [save_insts, restore_insts] = gen_save_insts(clobbered)

                call = asm.Instruction(inst.opcode, called_fn_arg)
                ctx.insts += [*save_insts, call, *restore_insts]

                log(call, free_regs)

                # Return any now-unused sources to the free set.
                for arg in args:
                    ctx.update_free_regs(arg, live_set)
                ctx.update_free_regs(called_fn, live_set)

                # Use the ABI-specified register for pulling out the return
                # value from the call. Move it to a fresh register so we can
                # keep it alive
                reg = ctx.instantiate_reg(RETURN_REGS[0])
                reg_assns[inst] = reg

            # Regular ops
            else:
                reg = None

                arg_regs = [ctx.get_arg_reg(arg, assign=True) for arg in inst.args]

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
                        reg = ctx.instantiate_reg(arg_regs[0])
                        log('destr copy', ctx.insts[-1], free_regs)
                        arg_regs[0] = reg
                    else:
                        reg = arg_regs[0]
                        log('clobber', inst.args[0], reg)
                        if inst.args[0] in reg_assns:
                            del reg_assns[inst.args[0]]
                            log('cl free instance')

                # Return any now-unused sources to the free set.
                for arg in inst.args:
                    ctx.update_free_regs(arg, live_set)

                # Non-destructive ops need a register assignment for their
                # implicit destination
                if not destructive and asm.needs_register(inst.opcode):
                    reg = ctx.alloc_reg(get_result_type(inst))
                    arg_regs = [reg] + arg_regs
                    log('nondestr assign', reg, free_regs)

                reg_assns[inst] = reg

                # And now return the destination of the instruction to the
                # free registers, although this only happens when the result
                # is never used...
                if inst not in live_set and asm.needs_register(inst.opcode):
                    log('free inst', inst, arg_regs[0])
                    ctx.dealloc_reg(arg_regs[0])

                ctx.insts.append(asm.Instruction(inst.opcode, *arg_regs))

                log('main', ctx.insts[-1], free_regs)

        # Remember allocated registers for the phi read
        regs = {name: reg_assns[inst] for [name, inst] in
                sorted(block.phi_read.args.items())}
        phi_assns[block.phi_read] = regs

        ctx.end_block(block)

    for block in fn.blocks:
        log('assns for', block.name, ctx.block_reg_assns[block])

    # Undefined variable check, should have a real message
    assert not fn.blocks[0].phi_write.args, fn.blocks[0].phi_write.args

    # Implement phis. For each CFG edge, we do a parallel move from the phi read
    # phase to the phi write phase, moving all registers that are needed in the
    # successor block (not all successors use all arguments of the read). The
    # actual move/swap instructions for the parallel move must go in either the
    # predecessor or successor, depending on which is unique (see
    # split_critical_edges() for an explanation how/why we ensure this is true)
    for block in fn.blocks:
        phi_w = phi_assns[block.phi_write]
        first = len(block.preds) != 1

        for pred in block.preds:
            phi_r = phi_assns[pred.phi_read]
            # Check types
            for key in phi_w:
                assert type(phi_w[key]) == type(phi_r[key]), (phi_w, phi_r)

            free_regs = ctx.free_regs_out[pred] if first else ctx.free_regs_in[block]

            insts = []
            for t in REG_TYPES:
                keys = [key for key in phi_w if phi_types[block][key] == t]
                ins = move_phi_args(phi_r, phi_w, keys, free_regs[t])
                for [inst, dst, src] in move_phi_args(phi_r, phi_w, keys, free_regs[t]):
                    if t == asm.GPReg:
                        if inst == 'mov':
                            insts.append(asm.Instruction('mov', dst, src))
                        else:
                            assert inst == 'swap'
                            insts.append(asm.Instruction('xchg', src, dst))
                    else:
                        if inst == 'mov':
                            insts.append(asm.Instruction('vmovdqu', dst, src))
                        else:
                            assert 0

            if first:
                ctx.block_insts[pred].extend(insts)
            else:
                ctx.block_insts[block][:0] = insts

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
        insts.extend(ctx.block_insts[block])

        # Add jump instructions
        for jmp in get_jumps(block, exit_phys_idx, exit_label):
            assert asm.is_jump_op(jmp.opcode)
            insts.append(asm.Instruction(jmp.opcode, *jmp.args))

    # Generate push/pop instructions for callee-save instructions, and
    # adjust the stack for phi
    clobbered = [reg for reg in reg_assns.values()
            if reg in VEC_CALLER_SAVE or reg in CALLER_SAVE]
    [save_insts, restore_insts] = gen_save_insts(
            clobbered, extra_stack=len(ctx.stack_assns)*8)

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
