import dumb_regalloc
import regalloc

class SSABasicBlock(phis, insts):
    pass

class Store(name, value):
    def __str__(self):
        return '{} = {}'.format(self.name, self.value)

class Inst(opcode, *args):
    def __str__(self):
        return '{}({})'.format(self.opcode, ', '.join(map(str, self.args)))

# Given a block that contains a list of top-level expression trees, flatten out
# the list so that every expression tree is only one ply deep, with back references to
# earlier parts of the list where branches were transplanted.
def flatten_block(block):
    @fixed_point
    def rec_flatten(rec_flatten, results, inst):
        if isinstance(inst, int):
            return [results + [['literal', inst]], len(results)]
        elif isinstance(inst, str):
            return [results, inst]
        elif isinstance(inst, Store):
            [results, ref] = rec_flatten(results, inst.value)
            return [results + [['store', inst.name, ref]], inst.name]
        else:
            args = []
            for arg in inst.args:
                [results, ref] = rec_flatten(results, arg)
                args = args + [ref]
            return [results + [[inst.opcode] + args], len(results)]

    results = []
    for inst in block.insts:
        [results, ref] = rec_flatten(results, inst)

    return results

def gen_ssa(blocks):
    exit_states = {}
    new_blocks = []
    for [block_id, block] in enumerate(blocks):
        phi_syms = set()
        current_syms = {}
        phis = []
        insts = []
        # Look for any symbol references in the blocks, and keep track of where they
        # are last assigned to
        for [i, inst] in enumerate(flatten_block(block)):
            [opcode, args] = [inst[0], inst[1:]]
            if opcode == 'store':
                [name, value] = args
                current_syms = current_syms + {name: i}
                # Placeholder to keep IDs the same after this
                insts = insts + [['dup', value]]
            else:
                new_args = []
                for arg in args:
                    if isinstance(arg, str):
                        if arg in current_syms:
                            new_args = new_args + [current_syms[arg]]
                        else:
                            new_args = new_args + [['phi', len(phis)]]
                            phis = phis + [arg]
                    else:
                        new_args = new_args + [arg]
                insts = insts + [[opcode] + new_args]
        new_blocks = new_blocks + [SSABasicBlock(phis, insts)]
        exit_states = exit_states + {block_id: current_syms}

    [preds, succs] = regalloc.get_block_linkage(new_blocks)

    # Fix up phis now that all blocks have been flattened
    blocks = new_blocks
    new_blocks = []
    for [block_id, block] in enumerate(blocks):
        insts = []
        phi_args = []
        for phi in block.phis:
            for pred in preds[block_id]:
                phi_args = phi_args + [[pred, exit_states[pred][phi] +
                    len(new_blocks[pred].phis)]]
            insts = insts + [['phi'] + phi_args]

        # Ugh, add an offset for each block's instruction IDs to account for all the phis
        for inst in block.insts:
            [opcode, args] = [inst[0], inst[1:]]
            if opcode != 'literal':
                args = [arg[1] if isinstance(arg, list) else
                    arg + len(block.phis) for arg in args]
            insts = insts + [[opcode] + args]
        new_blocks = new_blocks + [dumb_regalloc.BasicBlock(insts)]

    return new_blocks

def mov64(a):
    return Inst('mov64', a)

def add64(a, b):
    # Hacky instruction selection logic
    if isinstance(a, int):
        a = mov64(a)
    return Inst('add64', a, b)

blocks = [dumb_regalloc.BasicBlock([
        Store('x', mov64(1)),
        Store('x', add64(add64(add64('x', 2), 4), add64(mov64(4), 8))),
    ]),
]

dumb_regalloc.export_function('elfout.o', '_test', gen_ssa(blocks))
