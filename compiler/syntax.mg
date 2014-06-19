# XXX Obviously this file is just a stupid shell of an implementation

class BinaryOp(op, lhs, rhs, **k):
    def __str__(self): return '({} {} {})'.format(self.lhs, self.op, self.rhs)
class UnaryOp(*a,**k): pass
class Target(*a,**k): pass
class Identifier(name, **k):
    def __str__(self): return self.name
class None_(*a,**k): pass
class Boolean(*a,**k): pass
class String(*a,**k): pass
class Integer(value, **k):
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
class Return(*a,**k): pass
class Assert(*a,**k): pass
class Assignment(*a,**k): pass
class Block(*a,**k): pass
class For(*a,**k): pass
class While(*a,**k): pass
class IfElse(*a,**k): pass
class VarParams(*a,**k): pass
class KeywordVarParams(name, **k): pass
class Params(*a,**k): pass
class Function(*a,**k): pass
class Class(*a,**k): pass
class Import(*a,**k): pass
