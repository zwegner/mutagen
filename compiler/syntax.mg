class Target(targets,**k):
    def __str__(self):
        return self.targets[0]

def node(*edges):
    def xform(cls):
        # Parse edge specifications
        new_edges = []
        for attr in edges:
            if attr[0] in {'*', '#', '?'}:
                new_edges = new_edges + [[attr[1:], attr[0]]]
            else:
                new_edges = new_edges + [[attr, None]]

        attrs = {'_node_edges': new_edges}
        return hacky_class_from_base_and_new_attrs(cls.__name__, cls, attrs, cls.__params__)
    return xform

@node()
class Phi(name):
    def _str(self, graph):
        return 'Phi({})'.format(self.name)

@node('lhs', 'rhs')
class BinaryOp(op: str, lhs, rhs, **k):
    def _str(self, graph):
        return '({} {} {})'.format(self.lhs(graph)._str(graph), self.op, self.rhs(graph)._str(graph))

class UnaryOp(*a,**k): pass

@node()
class Parameter(index: int, **k):
    def _str(self, graph):
        return 'Parameter({})'.format(self.index)

@node()
class Identifier(name: str, **k):
    def _str(self, graph):
        return self.name

class None_(*a,**k): pass

@node()
class Boolean(value, **k):
    def _str(self, graph):
        return str(self.value)

@node()
class String(value, **k):
    def _str(self, graph):
        return str(self.value)

@node()
class Integer(value, **k):
    def _str(self, graph):
        return str(self.value)

class Scope(*a,**k): pass
class List(*a,**k): pass
class Dict(*a,**k): pass
class ListComprehension(*a,**k): pass
class DictComprehension(*a,**k): pass
class CompIter(*a,**k): pass
class GetItem(expr, item, **k): pass

class GetAttr(expr, attr, **k): pass

@node('fn', '*args')
class Call(fn, args, **k):
    def _str(self, graph):
        return '{}({})'.format(self.fn(graph)._str(graph), ', '.join(map(lambda (a): a._str(graph),
            self.args_gen(graph))))

class KeywordArg(*a,**k): pass
class VarArg(*a,**k): pass
class KeywordVarArg(*a,**k): pass
class Break(*a,**k): pass
class Continue(*a,**k): pass
class Yield(*a,**k): pass

@node('expr')
class Return(expr,**k):
    def _str(self, graph):
        return 'return {}'.format(self.expr(graph)._str(graph))

class Assert(*a,**k): pass

@node('value')
class Assignment(target, value, **k):
    def _str(self, graph):
        return '{} = {}'.format(self.target, self.value(graph)._str(graph))

@node('*stmts', '*phis', '*preds', '?test', '*succs', '#exit_states')
class BasicBlock(stmts=[], phis=[], preds=[], test=None, succs=[], exit_states={}):
    def _str(self, graph):
        return 'BBLOCK:\n' + '\n'.join(['  ' + stmt._str(graph) for stmt in self.stmts_gen(graph)])

@node('*stmts')
class Block(stmts, **k):
    def _str(self, graph):
        return 'BLOCK:\n' + '\n'.join(['  ' + stmt._str(graph) for stmt in self.stmts_gen(graph)])

class For(*a,**k): pass

@node('test', 'block')
class While(test, block, **k):
    pass

@node('test', 'if_block', 'else_block')
class IfElse(test, if_block, else_block, **k):
    pass

class VarParams(*a,**k): pass
class KeywordVarParams(name, **k): pass
class Params(*a,**k): pass
class Function(*a,**k): pass
class Class(*a,**k): pass
class Import(*a,**k): pass
