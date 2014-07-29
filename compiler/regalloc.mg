import asm
import elf

all_registers = set(range(16))

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
                for arg in inst[1:]:
                    live_set = live_set | {arg}

            # Make sure we aren't live before we exist
            live_set = live_set - {i}

        # Be sure to put the list back in forward order
        return list(reversed(live_sets))

class VirtualRegister(index: int):
    def __str__(self):
        return 'VReg({})'.format(self.index)

def get_liveness(blocks):
    block_outs = {block_id: {} for [block_id, block] in enumerate(blocks)}
    phi_reg_assns = {}
    n_phis = 0

    for [block_id, block] in enumerate(blocks):
        phi_regs = []
        for inst in block.insts:
            [opcode, args] = [inst[0], inst[1:]]
            if opcode == 'phi':
                reg = VirtualRegister(n_phis)
                n_phis = n_phis + 1
                phi_regs = phi_regs + [reg]
                # Go through all operands, which consist of a block and an
                # operand within that block, and make sure we add them to the
                # live outs of the proper block, and mark the assigned register.
                for [src_block_id, src_op] in args:
                    if src_op not in block_outs[src_block_id]:
                        outs = block_outs[src_block_id] + {src_op: reg}
                        block_outs = block_outs + {src_block_id: outs}

        phi_reg_assns = phi_reg_assns + {block_id: phi_regs}

    return [block_outs, phi_reg_assns]

# This is slower than necessary. Interesting case for syntax/state handling.
@fixed_point
def postorder_traverse(postorder_traverse, succs, start, used):
    if start not in used:
        used = used | {start}
        for succ in succs[start]:
            for b in postorder_traverse(succs, succ, used):
                used = used | {b}
                yield b
        yield start

# Dominance algorithm from http://www.cs.rice.edu/~keith/EMBED/dom.pdf
def get_block_dominance(start, preds, succs):
    # Get postorder traversal minus the first block
    postorder = list(postorder_traverse(succs, start, set()))
    post_id = {b: i for [i, b] in enumerate(postorder)}

    # Function to find the common dominator between two blocks
    def intersect(doms, b1, b2):
        while b1 != b2:
            while post_id[b1] < post_id[b2]:
                b1 = doms[b1]
            while post_id[b1] > post_id[b2]:
                b2 = doms[b2]
        return b1

    doms = {b: None for b in preds.keys()} + {start: start}
    changed = True
    while changed:
        changed = False
        for b in reversed(postorder[:-1]):
            [new_idom, rest] = [preds[b][0], preds[b][1:]]
            assert doms[new_idom] != None
            for p in rest:
                if doms[p] != None:
                    new_idom = intersect(doms, new_idom, p)
            if doms[b] != new_idom:
                doms = doms + {b: new_idom}
                changed = True
    return doms

def is_register(reg):
    return (isinstance(reg, asm.Register) or isinstance(reg, VirtualRegister))

def color_registers(blocks):
    new_blocks = []
    [block_outs, phi_reg_assns] = get_liveness(blocks)

    for [block_id, block] in enumerate(blocks):
        insts = []

        live_sets = block.get_live_sets(set(block_outs[block_id].keys()))

        # Now make a run through the instructions. Since we're assuming
        # spill/fill has been taken care of already, this can be done linearly.
        free_regs = all_registers
        reg_assns = {}
        for [i, [inst, live_set]] in enumerate(zip(block.insts, live_sets)):
            [opcode, args] = [inst[0], inst[1:]]
            # Special literal opcode: just a placeholder so we can differentiate
            # instruction indices and just literal ints/strs/whatever.
            if opcode == 'literal':
                reg_assns = reg_assns + {i: args[0]}
            elif opcode == 'phi':
                # Just use the virtual register assigned during liveness analysis.
                reg_assns = reg_assns + {i: phi_reg_assns[block_id][i]}
            elif asm.is_jump_op(opcode):
                insts = insts + [[opcode] + args]
            elif args:
                arg_regs = [reg_assns[i] for i in args]

                # Handle destructive ops first, which might need a move into
                # a new register. Do this before we return any registers to the
                # free set, since the inserted move comes before the instruction.
                destructive = (asm.is_destructive_op(opcode) and
                    is_register(arg_regs[0]))

                if destructive:
                    # See if we can clobber the register. If not, copy it
                    # into another register and change the assignment so
                    # later ops can see it.
                    # XXX This can be done in two ways, but moreover this
                    # should probably be done during scheduling with spill/fill.
                    # This also needs to interact with coalescing, when we have that.
                    if args[0] in live_set:
                        [free_regs, reg] = free_regs.pop()
                        reg = asm.Register(reg)
                        insts = insts + [['mov64', reg, arg_regs[0]]]
                        arg_regs = [reg] + arg_regs[1:]
                    else:
                        reg = arg_regs[0]

                # Return any now-unused sources to the free set.
                for [arg, src] in list(zip(args, arg_regs))[1:]:
                    if is_register(src) and arg not in live_set:
                        free_regs = {src.index} | free_regs

                if not destructive and asm.needs_register(opcode):
                    # Non-destructive ops need a register assignment for their
                    # implicit destination.
                    # First, check if this operand is a live out of the block,
                    # in which case the register is already assigned.
                    if i in block_outs[block_id]:
                        reg = block_outs[block_id][i]
                    else:
                        [free_regs, reg] = free_regs.pop()
                        reg = asm.Register(reg)
                    arg_regs = [reg] + arg_regs

                reg_assns = reg_assns + {i: reg}

                # And now return the destination of the instruction to the
                # free registers, although this only happens when the result
                # is never used...
                if i not in live_set:
                    free_regs = {arg_regs[0].index} | free_regs

                insts = insts + [[opcode] + arg_regs]
            else:
                insts = insts + [[opcode]]

        new_blocks = new_blocks + [BasicBlock(insts)]

    return new_blocks

def gen_insts(blocks):
    blocks = color_registers(blocks)
    insts = []

    for [block_id, block] in enumerate(blocks):
        insts = insts + [asm.Label('block{}'.format(block_id), False)]
        for [i, inst] in enumerate(block.insts):
            [opcode, args] = [inst[0], inst[1:]]
            if asm.is_jump_op(opcode):
                # Make sure only the last instruction is a control flow op
                assert i == len(block.insts) - 1
                # XXX take a flags argument, and make sure it's from the latest
                # flags-writing instruction
                [dest_block_id] = args
                dest = asm.Label('block{}'.format(dest_block_id), False)
                insts = insts + [asm.Instruction(opcode, dest)]
            else:
                insts = insts + [asm.Instruction(opcode, *args)]

    return insts

# Determine all predecessor and successor blocks
def get_block_linkage(blocks):
    preds = {i: [] for i in range(len(blocks))}
    succs = {}

    for [i, block] in enumerate(blocks):
        last_inst = block.insts[-1]
        [opcode, args] = [last_inst[0], last_inst[1:]]
        dests = []
        # Jump: add the destination block and the fall-through block
        if asm.is_jump_op(opcode):
            dests = args
            if opcode != 'jmp':
                assert i + 1 < len(blocks)
                dests = dests + [i + 1]
        # ...or just the fallthrough
        elif i + 1 < len(blocks):
            dests = [i + 1]

        # Link up blocks
        succs = succs + {i: dests}
        for dest in dests:
            preds = preds + {dest: preds[dest] + [i]}

    return [preds, succs]

blocks = [
    BasicBlock([
        ['literal', 11],
        ['mov64', 0],
        ['cmp64', 1, 0],
        ['je', 2],
    ]),
    BasicBlock([
        ['literal', 22],
        ['literal', 17],
        ['mov64', 0],
        ['mov64', 1],
        ['je', 1],
    ]),
    BasicBlock([
        ['phi', [0, 1], [1, 2]],
        ['phi', [0, 1], [1, 3]],
        ['literal', 33],
        ['add64', 0, 2],
        ['add64', 0, 2],
        ['add64', 1, 3],
        ['ret'],
    ]),
]

# "tests"
[preds, succs] = get_block_linkage(blocks)
assert preds == {0: [], 1: [0, 1], 2: [0, 1]}
assert succs == {0: [2, 1], 1: [1, 2], 2: []}

preds = {0: [], 1: [0], 2: [0], 3: [1, 4], 4: [2, 3]}
succs = {0: [1, 2], 1: [3], 2: [4], 3: [4], 4: [3]}
assert get_block_dominance(0, preds, succs) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
preds = {0: [], 1: [0], 2: [0], 3: [1, 4], 4: [2, 3, 5], 5: [2, 4]}
succs = {0: [1, 2], 1: [3], 2: [4, 5], 3: [4], 4: [3, 5], 5: [4]}
assert get_block_dominance(0, preds, succs) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
preds = {6: [], 5: [6], 4: [6], 1: [5, 2], 2: [4, 1, 3], 3: [4, 2]}
succs = {6: [4, 5], 5: [1], 4: [3, 2], 1: [2], 2: [1, 3], 3: [2]}
assert get_block_dominance(6, preds, succs) == {1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6}

insts = gen_insts(blocks)

for inst in insts:
    if isinstance(inst, asm.Label):
        print('{}:'.format(inst.name))
    else:
        print('    {}'.format(inst))

elf_file = elf.create_elf_file(*asm.build(insts))
write_binary_file('elfout.o', elf_file)
