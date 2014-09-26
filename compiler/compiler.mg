from dumb_regalloc import *
# XXX work around weird importing behavior
asm = regalloc.asm

class Parameter(index):
    pass

class PhiRef(index):
    def __repr__(self):
        return 'Phi({})'.format(self.index)

class Store(name, value):
    def __repr__(self):
        return '{} = {}'.format(self.name, self.value)

class Inst(opcode, *args):
    def __repr__(self):
        return '{}({})'.format(self.opcode, ', '.join(map(repr, self.args)))

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

    return block <- .insts = results

@fixed_point
def ensure_symbol_in_block(ensure_symbol_in_block, blocks, exit_states, preds,
    block_id, sym):
    if sym not in exit_states[block_id]:
        exit_states = exit_states <- [block_id][sym] = PhiRef(len(blocks[block_id].phis))
        blocks = blocks <- [block_id].phis += [sym]
        for pred in preds[blocks[block_id].name]:
            [blocks, exit_states] = ensure_symbol_in_block(blocks, exit_states,
                preds, pred, sym)
    return [blocks, exit_states]

def gen_ssa(fn):
    # Add the parameters to the first block.
    stmts = [Store(name, Parameter(i)) for [i, name] in enumerate(fn.parameters)]
    blocks = fn.blocks
    if stmts:
        blocks = [BasicBlock('prelude', [], stmts)] + blocks

    blocks = list(map(flatten_block, blocks))

    [preds, succs] = regalloc.get_block_linkage(blocks)

    exit_states = id_remap = {i: {} for i in range(len(blocks))}
    for [block_id, block] in enumerate(blocks):
        insts = []
        # Look for any symbol references in the blocks, and keep track of where they
        # are last assigned to
        for [i, inst] in enumerate(block.insts):
            [opcode, args] = [inst[0], inst[1:]]
            # For stores, we hackily remap IDs to simplify downstream processing
            if opcode == 'store':
                [name, value] = args
                if isinstance(value, str):
                    [blocks, exit_states] = ensure_symbol_in_block(blocks,
                        exit_states, preds, block_id, value)
                    exit_states = exit_states <- [block_id][name] = exit_states[block_id][value]
                    # XXX what to do with id_remap here?
                else:
                    exit_states = exit_states <- [block_id][name] = value
                    id_remap = id_remap <- [block_id][i] = id_remap[block_id][value]
            else:
                new_args = []
                for arg in args:
                    if isinstance(arg, str):
                        [blocks, exit_states] = ensure_symbol_in_block(blocks,
                            exit_states, preds, block_id, arg)
                        new_args = new_args + [exit_states[block_id][arg]]
                    else:
                        new_args = new_args + [arg]
                id_remap = id_remap <- [block_id][i] = len(insts)
                insts = insts + [[opcode] + new_args]
        blocks = blocks <- [block_id].insts = insts

    for [block_id, block] in enumerate(blocks):
        phis = [['phi'] + [[pred, exit_states[pred][phi]]
            for pred in preds[block.name]] for phi in block.phis]

        blocks = blocks <- [block_id].phis = phis

    # Ugh, add an offset for each block's instruction IDs to account for all the phis
    for [block_id, block] in enumerate(blocks):
        insts = []
        for phi in block.phis:
            [opcode, args] = [phi[0], phi[1:]]
            assert opcode == 'phi'
            args = [[pred, (arg.index if isinstance(arg, PhiRef) else
                id_remap[pred][arg] + len(blocks[pred].phis))] for [pred, arg] in args]
            insts = insts + [[opcode] + args]

        for inst in block.insts:
            [opcode, args] = [inst[0], inst[1:]]
            if opcode != 'literal':
                args = [arg.index if isinstance(arg, PhiRef) else (
                    arg if isinstance(arg, asm.Label) else
                    id_remap[block_id][arg] + len(block.phis)) for arg in args]
            insts = insts + [[opcode] + args]
        blocks = blocks <- [block_id].insts = insts

    return Function(fn.parameters, blocks)

def test64(a, b): return Inst('test64', a, b)
def jz(a): return Inst('jz', a)
def jnz(a): return Inst('jnz', a)
def jmp(a): return Inst('jmp', a)
def ret(a): return Inst('ret', a)
def mov64(a): return Inst('mov64', a)
def add64(a, b): return Inst('add64', a, b)
def sub64(a, b): return Inst('sub64', a, b)
def call(fn, *args): return Inst('call', fn, *args)
