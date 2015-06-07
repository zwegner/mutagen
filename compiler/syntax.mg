import compiler
import libgraph
# XXX work around weird importing behavior
asm = compiler.asm

def block_name(block_id):
    return 'block{}'.format(block_id)

def gen_test_block(graph, test, test_block_name, exit_block_name):
    [prelude, test] = test.gen_insts(graph)
    return [compiler.BasicBlock(test_block_name, [], prelude + [compiler.test64(test, test),
        compiler.jz(asm.LocalLabel(exit_block_name))])]

def node(*edges):
    def xform(cls):
        # Parse edge specifications
        new_edges = []
        for attr in edges:
            if attr[0] == '*':
                new_edges = new_edges + [[attr[1:], '*']]
            else:
                new_edges = new_edges + [[attr, None]]
        edge_names = [edge[0] for edge in new_edges]

        # Add a function to insert this node into a graph representation
        def add_to_graph(self, graph, new_class_dict):
            # XXX change once parameters are first class objects
            new_class = new_class_dict[type(self)]
            names = list(filter(lambda(name): name not in edge_names, new_class.__params__.names))
            args = [getattr(self, name) for name in names]

            node_id = len(graph.nodes)
            nobj = new_class(node_id, *args)
            graph = graph.add_node(nobj)

            for [attr, edge_type] in new_edges:
                if edge_type == '*':
                    for item in getattr(self, attr):
                        [graph, value] = item.add_to_graph(graph, new_class_dict)
                        graph = graph.append_to_edge(nobj, attr, value)
                else:
                    [graph, value] = getattr(self, attr).add_to_graph(graph, new_class_dict)
                    graph = graph.set_edge(nobj, attr, value)
            return [graph, nobj]

        attrs = {'add_to_graph': add_to_graph, '_node_edges': new_edges}
        return hacky_class_from_base_and_new_attrs(cls.__name__, cls, attrs, cls.__params__)
    return xform

def create_graph_class(cls):
    attrs = {'__name__': 'G' + cls.__name__}
    # Create accessor functions for each edge label
    for [attr, edge_type] in cls._node_edges:
        if edge_type == '*':
            attrs = attrs <- [attr] = lambda (self, graph, index): graph.get(self, attr)[index]
            attrs = attrs <- [attr + '_gen'] = def (self, graph) {
                for edge in graph.get(self, attr) {
                    yield edge;
                }
            }
            attrs = attrs <- [attr + '_set'] = lambda (self, graph, index, value): graph.set_edge_index(self, attr, index, value)
            attrs = attrs <- [attr + '_append'] = lambda (self, graph, value): graph.append_to_edge(self, attr, value)
        else:
            attrs = attrs <- [attr] = lambda (self, graph): graph.get(self, attr)
            attrs = attrs <- [attr + '_set'] = lambda (self, graph, value): graph.set(self, attr, value)

    # Adjust class parameters
    edge_names = [edge[0] for edge in cls._node_edges]
    params = cls.__params__
    names = ['node_id']
    types = [int]
    for [n, t] in zip(params.names, params.types):
        if n not in edge_names:
            names = names + [n]
            types = types + [t]
    params = Params(names, types, params.var_params,
        params.kw_params, params.kw_var_params)

    return hacky_class_from_base_and_new_attrs('G' + cls.__name__, cls, attrs, params)

@node('lhs', 'rhs')
class BinaryOp(op: str, lhs, rhs, **k):
    def gen_insts(self, graph):
        [lhs_insts, lhs] = self.lhs(graph).gen_insts(graph)
        [rhs_insts, rhs] = self.rhs(graph).gen_insts(graph)
        fn = {
            '+': compiler.add64,
            '-': compiler.sub64,
            '&': compiler.and64,
            '|': compiler.or64,
        }[self.op]
        return [lhs_insts + rhs_insts, fn(lhs, rhs)]
    def __str__(self):
        return '({} {} {})'.format(self.lhs, self.op, self.rhs)

class UnaryOp(*a,**k): pass

@node()
class Target(targets,**k):
    def __str__(self):
        return self.targets[0]

@node()
class Identifier(name: str, **k):
    def gen_insts(self, graph):
        return [[], self.name]
    def __str__(self):
        return self.name

class None_(*a,**k): pass

@node()
class Boolean(value, **k):
    def gen_insts(self, graph):
        assert False
    def __str__(self):
        return str(self.value)

@node()
class String(value, **k):
    def gen_insts(self, graph):
        assert False
    def __str__(self):
        return str(self.value)

@node()
class Integer(value, **k):
    def gen_insts(self, graph):
        return [[], compiler.mov64(self.value)]
    def __str__(self):
        return str(self.value)

class Scope(*a,**k): pass
class List(*a,**k): pass
class Dict(*a,**k): pass
class ListComprehension(*a,**k): pass
class DictComprehension(*a,**k): pass
class CompIter(*a,**k): pass
class GetItem(expr, item, **k): pass

class GetAttr(expr, attr, **k):
    def __str__(self):
        return '{}.{}'.format(self.expr, self.attr)

@node('fn', '*args')
class Call(fn, args, **k):
    def gen_insts(self, graph):
        # Stupid temporary hack
        fn = self.fn(graph).name
        if fn in {'test', 'deref', 'atoi'}:
            fn = asm.ExternLabel('_' + fn)
        [insts, args] = list(zip(*[arg.gen_insts(graph) for arg in self.args_gen(graph)]))
        insts = sum(insts, [])
        return [insts, compiler.call(fn, *args)]
    def __str__(self):
        return '{}({})'.format(self.fn, ', '.join(map(str, self.args)))

class KeywordArg(*a,**k): pass
class VarArg(*a,**k): pass
class KeywordVarArg(*a,**k): pass
class Break(*a,**k): pass
class Continue(*a,**k): pass
class Yield(*a,**k): pass

@node('expr')
class Return(expr,**k):
    def gen_insts(self, graph):
        [insts, ref] = self.expr(graph).gen_insts(graph)
        return [insts, compiler.ret(ref)]

class Assert(*a,**k): pass

@node('target', 'value')
class Assignment(target, value, **k):
    def gen_insts(self, graph):
        [name] = self.target(graph).targets
        assert isinstance(name, str)
        [insts, ref] = self.value(graph).gen_insts(graph)
        return [insts, compiler.Store(name, ref)]
    def __str__(self):
        return '{} = {}'.format(self.target, self.value)

@node('*stmts')
class Block(stmts, **k):
    def gen_blocks(self, graph, block_id):
        blocks = []
        current_block_insts = []
        for stmt in self.stmts_gen(graph):
            if hasattr(stmt, 'gen_blocks'):
                if current_block_insts:
                    blocks = blocks + [compiler.BasicBlock(block_name(block_id), [], current_block_insts)]
                    block_id = block_id + 1
                    current_block_insts = []
                [child_blocks, block_id] = stmt.gen_blocks(graph, block_id)
                blocks = blocks + child_blocks
            else:
                [i, r] = stmt.gen_insts(graph)
                current_block_insts = current_block_insts + i + [r]

        if current_block_insts:
            blocks = blocks + [compiler.BasicBlock(block_name(block_id), [], current_block_insts)]
            block_id = block_id + 1
        return [blocks, block_id]

class For(*a,**k): pass

@node('test', 'block')
class While(test, block, **k):
    def gen_blocks(self, graph, block_id):
        test_block_name = block_name(block_id)
        [while_blocks, block_id] = self.block(graph).gen_blocks(graph, block_id + 1)
        post_block_name = block_name(block_id)
        block_id = block_id + 1
        # HACK: assume another block comes after this...
        exit_block_name = block_name(block_id)

        blocks = gen_test_block(graph, self.test(graph), test_block_name, exit_block_name)
        blocks = blocks + while_blocks
        blocks = blocks + [compiler.BasicBlock(post_block_name, [], [
            compiler.jmp(asm.LocalLabel(test_block_name))])]
        return [blocks, block_id]

class IfElse(test, if_block, else_block, **k):
    def gen_blocks(self, graph, block_id):
        test_block_name = block_name(block_id)
        [if_blocks, block_id] = self.if_block.gen_blocks(graph, block_id + 1)
        post_block_name = block_name(block_id)

        block_id = block_id + 1
        else_block_name = block_name(block_id)
        [else_blocks, block_id] = self.else_block.gen_blocks(graph, block_id)

        # HACK: assume another block comes after this...
        exit_block_name = block_name(block_id)

        blocks = gen_test_block(graph, self.test, test_block_name, else_block_name)
        blocks = blocks + if_blocks
        blocks = blocks + [compiler.BasicBlock(post_block_name, [], [
            compiler.jmp(asm.LocalLabel(exit_block_name))])]
        blocks = blocks + else_blocks
        return [blocks, block_id]

class VarParams(*a,**k): pass
class KeywordVarParams(name, **k): pass
class Params(*a,**k): pass
class Function(*a,**k): pass
class Class(*a,**k): pass
class Import(*a,**k): pass

def graphulate(tree):
    # Ugh
    new_class_dict = {cls: create_graph_class(cls) for cls in [BinaryOp, Integer, Identifier,
        Target, Assignment, Block, While, Return, Call]}

    graph = libgraph.DirectedGraph()
    [graph, nt] = tree.add_to_graph(graph, new_class_dict)
    return [graph, nt]

def gen_insts(stmts):
    block = Block(stmts)
    [graph, block] = graphulate(block)
    [blocks, block_id] = block.gen_blocks(graph, 0)
    fn = compiler.Function(['argc', 'argv'], blocks)
    return {'_main': compiler.gen_ssa(fn)}
