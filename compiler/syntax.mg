# XXX Obviously this file is just a stupid shell of an implementation

import compiler
# XXX work around weird importing behavior
asm = compiler.asm

def block_name(block_id):
    return 'block{}'.format(block_id)

def gen_blocks_from_stmts(stmts, block_id):
    blocks = []
    current_block_insts = []
    for stmt in stmts:
        if hasattr(stmt, 'gen_blocks'):
            if current_block_insts:
                blocks = blocks + [compiler.BasicBlock(block_name(block_id), [], current_block_insts)]
                block_id = block_id + 1
                current_block_insts = []
            [child_blocks, block_id] = stmt.gen_blocks(block_id)
            blocks = blocks + child_blocks
        else:
            [i, r] = stmt.gen_insts()
            current_block_insts = current_block_insts + i + [r]

    if current_block_insts:
        blocks = blocks + [compiler.BasicBlock(block_name(block_id), [], current_block_insts)]
        block_id = block_id + 1
    return [blocks, block_id]

def gen_test(value):
    return compiler.test64(value, value)

class BinaryOp(op, lhs, rhs, **k):
    def gen_insts(self):
        [lhs_insts, lhs] = self.lhs.gen_insts()
        [rhs_insts, rhs] = self.rhs.gen_insts()
        fn = {'+': compiler.add64, '-': compiler.sub64}[self.op]
        return [lhs_insts + rhs_insts, fn(lhs, rhs)]
    def __str__(self): return '({} {} {})'.format(self.lhs, self.op, self.rhs)
class UnaryOp(*a,**k): pass
class Target(targets,**k):
    def __str__(self): return self.targets[0]
class Identifier(name, **k):
    def gen_insts(self):
        return [[], self.name]
    def __str__(self): return self.name
class None_(*a,**k): pass
class Boolean(*a,**k): pass
class String(*a,**k): pass
class Integer(value, **k):
    def gen_insts(self):
        return [[], compiler.mov64(self.value)]
    def __str__(self): return str(self.value)
class Scope(*a,**k): pass
class List(*a,**k): pass
class Dict(*a,**k): pass
class ListComprehension(*a,**k): pass
class DictComprehension(*a,**k): pass
class CompIter(*a,**k): pass
class GetItem(expr, item, **k): pass
class GetAttr(expr, attr, **k):
    def __str__(self): return '{}.{}'.format(self.expr, self.attr)
class Call(fn, args, **k):
    def __str__(self): return '{}({})'.format(self.fn, ', '.join(map(str, self.args)))
class KeywordArg(*a,**k): pass
class VarArg(*a,**k): pass
class KeywordVarArg(*a,**k): pass
class Break(*a,**k): pass
class Continue(*a,**k): pass
class Yield(*a,**k): pass
class Return(expr,**k):
    def gen_insts(self):
        [insts, ref] = self.expr.gen_insts()
        return [insts, compiler.ret(ref)]
class Assert(*a,**k): pass
class Assignment(target, value, **k):
    def gen_insts(self):
        [name] = self.target.targets
        assert isinstance(name, str)
        [insts, ref] = self.value.gen_insts()
        return [insts, compiler.Store(name, ref)]
    def __str__(self):
        return '{} = {}'.format(self.target, self.value)
class Block(stmts, **k): pass
class For(*a,**k): pass
class While(test, block, **k):
    def gen_blocks(self, block_id):
        pre_block = block_name(block_id)
        [while_blocks, block_id] = gen_blocks_from_stmts(self.block.stmts, block_id + 1)
        post_block = block_name(block_id)
        block_id = block_id + 1
        # HACK: assume another block comes after this...
        exit_block = block_name(block_id)

        [prelude, test] = self.test.gen_insts()
        blocks = [compiler.BasicBlock(pre_block, [], prelude + [gen_test(test),
            compiler.jz(asm.LocalLabel(exit_block))])] + while_blocks + [
            compiler.BasicBlock(post_block, [], [compiler.jmp(asm.LocalLabel(pre_block))])]
        return [blocks, block_id]

class IfElse(*a,**k): pass
class VarParams(*a,**k): pass
class KeywordVarParams(name, **k): pass
class Params(*a,**k): pass
class Function(*a,**k): pass
class Class(*a,**k): pass
class Import(*a,**k): pass

def gen_insts(stmts):
    [blocks, block_id] = gen_blocks_from_stmts(stmts, 0)
    fn = compiler.Function(['argc', 'argv'], blocks)
    return {'_main': compiler.gen_ssa(fn)}
