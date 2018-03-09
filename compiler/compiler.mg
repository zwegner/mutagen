import dumb_regalloc
# XXX work around weird importing behavior
asm = dumb_regalloc.asm
regalloc = dumb_regalloc.regalloc
lir = dumb_regalloc.lir

# XXX change this interface once basic block handling stuff is in here
def compile(parameters, basic_blocks, node_map):
    return dumb_regalloc.Function(parameters, basic_blocks, node_map)
