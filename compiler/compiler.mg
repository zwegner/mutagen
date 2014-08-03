import dumb_regalloc
# XXX work around weird importing behavior
regalloc = dumb_regalloc.regalloc
asm = regalloc.asm

class SSABasicBlock(phis, insts):
    pass

class Parameter(index):
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
        elif isinstance(inst, Parameter):
            return [results + [['parameter', inst.index]], len(results)]
        elif isinstance(inst, regalloc.asm.Label):
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

def gen_ssa(fn):
    # Add the parameters to the first block.
    # HACK: add them to the first block instead of a new block since our liveness analysis sucks
    stmts = [Store(name, Parameter(i)) for [i, name] in enumerate(fn.parameters)] + fn.blocks[0].insts
    blocks = [dumb_regalloc.BasicBlock(stmts)] + fn.blocks[1:]

    id_remap = {i: {} for i in range(len(blocks))}
    exit_states = {}
    new_blocks = []
    for [block_id, block] in enumerate(blocks):
        current_syms = {}
        phis = []
        insts = []
        # Look for any symbol references in the blocks, and keep track of where they
        # are last assigned to
        for [i, inst] in enumerate(flatten_block(block)):
            [opcode, args] = [inst[0], inst[1:]]
            if opcode == 'store':
                [name, value] = args
                current_syms = current_syms + {name: value}
                id_remap = id_remap <- [block_id][i] = id_remap[block_id][value]
            else:
                new_args = []
                for arg in args:
                    if isinstance(arg, str):
                        if arg in current_syms:
                            new_args = new_args + [current_syms[arg]]
                        elif arg in phis:
                            new_args = new_args + [['phi', phis.index(arg)]]
                        else:
                            new_args = new_args + [['phi', len(phis)]]
                            phis = phis + [arg]
                    else:
                        new_args = new_args + [arg]
                id_remap = id_remap <- [block_id][i] = len(insts)
                insts = insts + [[opcode] + new_args]
        new_blocks = new_blocks + [SSABasicBlock(phis, insts)]
        exit_states = exit_states + {block_id: current_syms}

    [preds, succs] = regalloc.get_block_linkage(new_blocks)

    # Fix up phis now that all blocks have been flattened
    blocks = new_blocks
    new_blocks = []
    for [block_id, block] in enumerate(blocks):
        insts = []
        for phi in block.phis:
            phi_args = []
            for pred in preds[block_id]:
                src_inst = id_remap[pred][exit_states[pred][phi]] + len(blocks[pred].phis)
                phi_args = phi_args + [[pred, src_inst]]
            insts = insts + [['phi'] + phi_args]

        # Ugh, add an offset for each block's instruction IDs to account for all the phis
        for inst in block.insts:
            [opcode, args] = [inst[0], inst[1:]]
            if opcode != 'literal':
                args = [arg[1] if isinstance(arg, list) else (
                    arg if isinstance(arg, asm.Label) else
                    id_remap[block_id][arg] + len(block.phis)) for arg in args]
            insts = insts + [[opcode] + args]
        new_blocks = new_blocks + [dumb_regalloc.BasicBlock(insts)]

    return dumb_regalloc.Function(fn.parameters, new_blocks)

def mov64(a):
    return Inst('mov64', a)

def jnz(a):
    return Inst('jnz', a)

def ret(a):
    return Inst('ret', a)

def add64(a, b):
    return Inst('add64', a, b)

def sub64(a, b):
    return Inst('sub64', a, b)

fn = dumb_regalloc.Function(['count'], [
    dumb_regalloc.BasicBlock([
        Store('x', mov64(0)),
    ]),
    dumb_regalloc.BasicBlock([
        Store('x', add64('x', 'count')),
        Store('count', sub64('count', 1)),
        jnz(asm.Label(1, False)),
    ]),
    dumb_regalloc.BasicBlock([
        ret('x'),
    ]),
])

dumb_regalloc.export_function('elfout.o', '_test', gen_ssa(fn))
