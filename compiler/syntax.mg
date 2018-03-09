import compiler
import libgraph
# XXX work around weird importing behavior
asm = compiler.asm
dumb_regalloc = compiler.dumb_regalloc
lir = compiler.lir

def block_name(block_id):
    return '$block{}'.format(block_id)

def hacky_type_check(obj, type_name):
    return type(obj).__name__ == 'G' + type_name

# Walk the CFG in preorder
def walk_blocks(graph, first):
    seen = {}
    work = [first]
    while work:
        [block, work] = [work[0], work[1:]]
        yield block
        seen = seen <- [block] = 1
        for succ in block.succs_gen(graph):
            if succ not in seen:
                work = work + [succ]

class Target(targets,**k):
    def __str__(self):
        return self.targets[0]

class SSAContext(new_class_dict: dict, current_block=None, statements=[], simplified_set: dict={}):
    def load(self, graph, name):
        if name in self.current_block.exit_states(graph):
            return [graph, self.current_block.exit_states(graph)[name]]
        else:
            [graph, phi] = self.create_node(graph, 'Phi', name)
            graph = graph.append_to_edge(self.current_block, 'phis', phi)
            graph = graph.set_edge_key(self.current_block, 'exit_states', name, phi)
            return [graph, phi]

    def store(self, graph, name, value):
        graph = graph.set_edge_key(self.current_block, 'exit_states', name, value)
        return [graph, self]

    # XXX this is a hacky api
    def get_graph_class(self, cls):
        if not isinstance(cls, str):
            cls = cls.__name__
        return self.new_class_dict[cls]

    def create_node(self, graph, new_class, *args, **kwargs):
        if isinstance(new_class, str):
            new_class = self.get_graph_class(new_class)
        node_id = graph.node_seq_id
        node = new_class(node_id, *args, **kwargs)
        graph = graph.add_node(node) <- .node_seq_id += 1
        for [attr, edge_type] in new_class._node_edges:
            if edge_type == '*':
                graph = graph.create_edge_list(node, attr)
            elif edge_type == '#':
                graph = graph.create_edge_dict(node, attr)
        return [graph, node]

    def is_simplified(self, node):
        return node in self.simplified_set

    def add_to_simplified_set(self, node):
        return self <- .simplified_set[node] = 1

def node(*edges):
    def xform(cls):
        # Parse edge specifications
        new_edges = []
        for attr in edges:
            if attr[0] in {'*', '#', '?'}:
                new_edges = new_edges + [[attr[1:], attr[0]]]
            else:
                new_edges = new_edges + [[attr, None]]

        # Add a function to insert this node into a graph representation
        def add_to_graph(self, graph, ctx):
            new_class = ctx.get_graph_class(type(self))

            # XXX ughhhh... pick out all the non-node arguments to pass to the create_node call
            args = []
            for param in cls.__params__.names:
                if not any([attr == param for [attr, edge_type] in new_edges]):
                    args = args + [getattr(self, param)]

            [graph, node] = ctx.create_node(graph, new_class, *args)

            for [attr, edge_type] in new_edges:
                if edge_type == '*':
                    for item in getattr(self, attr):
                        [graph, value] = item.add_to_graph(graph, ctx)
                        graph = graph.append_to_edge(node, attr, value)
                elif edge_type == '#':
                    for [key, value] in getattr(self, attr):
                        #[graph, key] = key.add_to_graph(graph, ctx)
                        [graph, value] = value.add_to_graph(graph, ctx)
                        graph = graph.set_edge_key(node, key, value)
                else:
                    value = getattr(self, attr)
                    if edge_type != '?' or value != None:
                        [graph, value] = value.add_to_graph(graph, ctx)
                        graph = graph.set_edge(node, attr, value)
            return [graph, node]

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
            attrs = attrs <- [attr + '_append'] = lambda (self, graph, value): graph.append_to_edge(self, attr, value)
        elif edge_type == '?':
            # XXX HACK
            attrs = attrs <- [attr] = lambda (self, graph): graph.get(self, attr) if attr in graph.edge_sets[self] else None
        else:
            attrs = attrs <- [attr] = lambda (self, graph): graph.get(self, attr)

    def iterate_children(self, graph):
        for [attr, edge_type] in cls._node_edges:
            if edge_type == '*':
                for child in getattr(self, attr + '_gen')(graph):
                    yield child
            elif edge_type == '#':
                for [key, value] in getattr(self, attr)(graph):
                    #yield key
                    yield value
            else:
                child = getattr(self, attr)(graph)
                if edge_type != '?' or child != None:
                    yield child

    attrs = attrs + {'iterate_children': iterate_children}

    # Adjust class parameters
    edge_names = [edge[0] for edge in cls._node_edges]
    params = cls.__params__
    names = ['node_id']
    types = [int]
    for [n, t] in zip(params.names, params.types):
        if n not in edge_names:
            names = names + [n]
            types = types + [t]
    kwparams = []
    for kwp in params.kw_params:
        if kwp.name not in edge_names:
            kw_params = kw_params + [kwp]
    params = Params(names, types, params.var_params, kwparams, params.kw_var_params)

    return hacky_class_from_base_and_new_attrs('G' + cls.__name__, cls, attrs, params)

# rec_gen_ssa recursively walks the nodes in an AST, transforming identifiers/assignments so as to
# wire together a DAG of expressions. At this point in the compilation process, all blocks are
# handled independently, but two data structures are created here for linking the blocks together
# later: a list of phis, which are identifiers that are loaded in the block without being assigned
# to earlier in the block (and therefore must be set in every predecessor block in a well-formed
# program), and a dictionary of exit states, which contains the last value assigned to each variable
# within this block. In addition to this DAG creation, the expression trees are flattened, i.e. all
# nodes in the DAG are part of the statements list within their containing block. This aids later
# compilation passes by transforming a recursive DAG walk (like this one) into a linear scan.
@fixed_point
def rec_gen_ssa(rec_gen_ssa, node, graph, ctx):
    # Run gen_ssa on children
    for child in node.iterate_children(graph):
        [graph, ctx, child] = rec_gen_ssa(child, graph, ctx)
        # Flattening: if this child was a regular node (i.e. wasn't an identifier/assignment),
        # add it to the temporary statements list in the context. BasicBlock.gen_ssa() will
        # interleave these statements with the nodes that are already there, the main statements.
        if child != None and child.node_id not in ctx.simplified_set:
            ctx = (ctx <- .statements += [child], .simplified_set[child.node_id] = 1)

    # Transform identifiers/assignments by loading/storing to the exit_states dictionary
    if hacky_type_check(node, 'Identifier'):
        [graph, value] = ctx.load(graph, node.name)
        graph = graph.replace_node(node, value)
        node = None
    elif hacky_type_check(node, 'Assignment'):
        # XXX Only handle a single target for now. Compound targets, as well as any variable stores
        # that aren't part of a normal assignment (like in a for loop) should probably be handled
        # by macros.
        [target] = node.target.targets
        [graph, ctx] = ctx.store(graph, target, node.value(graph))
        node = None
    return [graph, ctx, node]

# Next pass: walk over the blocks and convert the SSA DAG to LIR (low-level intermediate representation).
# All child nodes in this representation are replaced with NR (node reference) objects, which just hold
# the node id of the child. We also do things like add jumps between basic blocks.
def generate_lir(graph, first_block):
    node_map = {}
    new_blocks = []
    for block in walk_blocks(graph, first_block):
        # Data structures for this block
        instructions = []
        phis = []

        # Make wacky phi placeholders that the register allocator can understand
        for phi in block.phis_gen(graph):
            # Look up which nodes are assigned to the variable that this phi tracks at the end
            # of each predecessor block
            args = [pred.exit_states(graph)[phi.name].node_id for pred in block.preds_gen(graph)]
            phis = phis + [lir.phi(phi.name, args)]

            # Update the node map. This is sorta weird since we're storing the phi name (a string)
            # instead of the instruction index (an int)
            node_map = node_map <- [phi.node_id] = [block_name(block.node_id), phi.name]

        for stmt in block.stmts_gen(graph):
            assert hasattr(stmt, 'gen_insts')
            inst = stmt.gen_insts(graph)
            # Update the node map and instruction list
            node_map = node_map <- [stmt.node_id] = [block_name(block.node_id), len(instructions)]
            instructions = instructions + [inst]

        test = block.test(graph)
        if test:
            # Sanity check, make sure the test has already had its instructions generated (it should
            # also be in the list of statements)
            assert test.node_id in node_map

            [if_block, else_block] = [asm.LocalLabel(block_name(succ.node_id)) for succ in block.succs_gen(graph)]
            # XXX for now use two jumps, don't rely on physical ordering
            instructions = instructions + [
                lir.test64(lir.NR(test.node_id), lir.NR(test.node_id)),
                lir.jnz(if_block),
                lir.jmp(else_block)
            ]
        else:
            # XXX for now add an unconditional jump to the fallthrough block, don't rely on physical ordering
            if len(list(block.succs_gen(graph))) == 1:
                succ = asm.LocalLabel(block_name(block.succs(graph, 0).node_id))
                instructions = instructions + [lir.jmp(succ)]
            else:
                assert len(list(block.succs_gen(graph))) == 0

        new_blocks = new_blocks + [dumb_regalloc.BasicBlock(block_name(block.node_id), phis, instructions)]

    return [new_blocks, node_map]

@node()
class Phi(name):
    def gen_insts(self, graph):
        return lir.phi_ref(self.name)
    def _str(self, graph):
        return 'Phi({})'.format(self.name)

binop_table = {
    '+': ['__add__', lir.add64],
    '-': ['__sub__', lir.sub64],
    '&': ['__and__', lir.and64],
    '|': ['__or__',  lir.or64],
}

@node('lhs', 'rhs')
class BinaryOp(op: str, lhs, rhs, **k):
    def simplify(self, graph, ctx):
        if (hacky_type_check(self.lhs(graph), 'Integer') and
                hacky_type_check(self.rhs(graph), 'Integer')):
            [lhs, rhs] = [self.lhs(graph).value, self.rhs(graph).value]
            # Urgh
            if self.op == '+':
                result = lhs + rhs
            elif self.op == '-':
                result = lhs - rhs
            elif self.op == '&':
                result = lhs & rhs
            elif self.op == '|':
                result = lhs | rhs
            else:
                print(self.op)
                assert False
            [graph, node] = ctx.create_node(graph, 'Integer', result)
            graph = graph.replace_node(self, node)
            return [graph, ctx, node]
        return [graph, ctx, self]
    def gen_insts(self, graph):
        fn = binop_table[self.op][1]
        return fn(lir.NR(self.lhs(graph).node_id), lir.NR(self.rhs(graph).node_id))
    def _str(self, graph):
        return '({} {} {})'.format(self.lhs(graph)._str(graph), self.op, self.rhs(graph)._str(graph))

class UnaryOp(*a,**k): pass

@node()
class Parameter(index: int, **k):
    def gen_insts(self, graph):
        return lir.parameter(self.index)
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
    def gen_insts(self, graph):
        return lir.mov64(self.value)
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
    def gen_insts(self, graph):
        # Stupid temporary hack
        fn = self.fn(graph).name
        if fn in {'test', 'test2', 'deref', 'atoi'}:
            fn = asm.ExternLabel('_' + fn)

        return lir.call(fn, *[lir.NR(arg.node_id) for arg in self.args_gen(graph)])
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
    def gen_insts(self, graph):
        return lir.ret(lir.NR(self.expr(graph).node_id))
    def _str(self, graph):
        return 'return {}'.format(self.expr(graph)._str(graph))

class Assert(*a,**k): pass

@node('value')
class Assignment(target, value, **k):
    def _str(self, graph):
        return '{} = {}'.format(self.target, self.value(graph)._str(graph))

@node('*stmts', '*phis', '*preds', '?test', '*succs', '#exit_states')
class BasicBlock(stmts=[], phis=[], preds=[], test=None, succs=[], exit_states={}):
    def gen_ssa(self, graph, ctx):
        new_stmts = []
        for stmt in self.stmts_gen(graph):
            ctx = ctx <- .statements = []

            [graph, ctx, new_stmt] = rec_gen_ssa(stmt, graph, ctx)
            new_stmts = new_stmts + ctx.statements
            if new_stmt != None:
                new_stmts = new_stmts + [new_stmt]

        # XXX unnecessary, test is added to the statements of each block
        #if self.test(graph) != None:
        #    ctx = ctx <- .statements = []
        #    [graph, ctx, new_stmt] = rec_gen_ssa(self.test(graph), graph, ctx)
        #    assert new_stmt != None
        #    new_stmts = new_stmts + ctx.statements

        graph = graph.set_edge_list(self, 'stmts', new_stmts)
        return [graph, ctx]
    def simplify(self, graph, ctx):
        if self.test(graph) != None:
            [graph, ctx, _] = self.test(graph).rec_simplify(graph, ctx)
        for stmt in self.stmts_gen(graph):
            [graph, ctx, _] = stmt.rec_simplify(graph, ctx)
        return [graph, ctx]
    def _str(self, graph):
        return 'BBLOCK:\n' + '\n'.join(['  ' + stmt._str(graph) for stmt in self.stmts_gen(graph)])

def link_blocks(graph, pred, succ):
    graph = pred.succs_append(graph, succ)
    graph = succ.preds_append(graph, pred)
    return graph

@node('*stmts')
class Block(stmts, **k):
    def gen_blocks(self, graph, ctx):
        [graph, prelude] = ctx.create_node(graph, 'BasicBlock')
        curr_block = prelude
        for stmt in self.stmts_gen(graph):
            if hasattr(stmt, 'gen_blocks'):
                [graph, first, last] = stmt.gen_blocks(graph, ctx)
                graph = link_blocks(graph, curr_block, first)
                [graph, curr_block] = ctx.create_node(graph, 'BasicBlock')
                graph = link_blocks(graph, last, curr_block)
            else:
                graph = curr_block.stmts_append(graph, stmt)
        return [graph, prelude, curr_block]
    def _str(self, graph):
        return 'BLOCK:\n' + '\n'.join(['  ' + stmt._str(graph) for stmt in self.stmts_gen(graph)])

class For(*a,**k): pass

@node('test', 'block')
class While(test, block, **k):
    def gen_blocks(self, graph, ctx):
        # Create blocks
        [graph, test_block] = ctx.create_node(graph, 'BasicBlock')
        test = self.test(graph)
        if test != None:
            graph = graph.set_edge_list(test_block, 'stmts', [test])
            graph = graph.set_edge(test_block, 'test', test)
        [graph, first, last] = self.block(graph).gen_blocks(graph, ctx)
        [graph, exit_block] = ctx.create_node(graph, 'BasicBlock')

        # Link them up
        graph = link_blocks(graph, test_block, first)
        graph = link_blocks(graph, test_block, exit_block)
        graph = link_blocks(graph, last, test_block)
        return [graph, test_block, exit_block]

@node('test', 'if_block', 'else_block')
class IfElse(test, if_block, else_block, **k):
    def gen_blocks(self, graph, ctx):
        # Create blocks
        [graph, test_block] = ctx.create_node(graph, 'BasicBlock')
        test = self.test(graph)
        graph = graph.set_edge_list(test_block, 'stmts', [test])
        graph = graph.set_edge(test_block, 'test', test)
        [graph, if_first, if_last] = self.if_block(graph).gen_blocks(graph, ctx)
        [graph, else_first, else_last] = self.else_block(graph).gen_blocks(graph, ctx)
        [graph, exit_block] = ctx.create_node(graph, 'BasicBlock')

        # Link them up
        graph = link_blocks(graph, test_block, if_first)
        graph = link_blocks(graph, test_block, else_first)
        graph = link_blocks(graph, if_last, exit_block)
        graph = link_blocks(graph, else_last, exit_block)
        return [graph, test_block, exit_block]

class VarParams(*a,**k): pass
class KeywordVarParams(name, **k): pass
class Params(*a,**k): pass
class Function(*a,**k): pass
class Class(*a,**k): pass
class Import(*a,**k): pass

def graphulate(block):
    # Ugh
    new_class_dict = {cls.__name__: create_graph_class(cls) for cls in [Phi, BinaryOp, Integer, Identifier,
        Parameter, Assignment, BasicBlock, Block, While, Return, Call]}
    ctx = SSAContext(new_class_dict)

    graph = libgraph.DirectedGraph()
    [graph, new_block] = block.add_to_graph(graph, ctx)

    return [graph, ctx, new_block]

def print_blocks(graph, block):
    for block in walk_blocks(graph, block):
        print('Block', block.node_id)
        print('  preds:', ' '.join([str(pred.node_id) for pred in block.preds_gen(graph)]))
        print('  succs:', ' '.join([str(succ.node_id) for succ in block.succs_gen(graph)]))
        print('  phis:')
        for phi in block.phis_gen(graph):
            print('    ', phi._str(graph))
        print('  stmts:')
        for stmt in block.stmts_gen(graph):
            print('    ', stmt._str(graph))
        print('    exit:', block.exit_states(graph))

def propagate_phis(ctx, graph, first_block):
    # Iterate through the graph and add phis to blocks whose successors load from a given
    # variable before they are assigned, which means
    # XXX This is WRONG, we're only propagating backwards one level
    for block in reversed(list(walk_blocks(graph, first_block))):
        for pred in block.preds_gen(graph):
            for phi in block.phis_gen(graph):
                if phi.name not in pred.exit_states(graph):
                    [graph, phi_node] = ctx.create_node(graph, 'Phi', phi.name)
                    graph = graph.append_to_edge(pred, 'phis', phi_node)
                    graph = graph.set_edge_key(pred, 'exit_states', phi.name, phi_node)

    return graph

def compile(stmts):
    # Add statements for parameter handling
    # XXX move this to compiler.mg once all the basic block stuff is there
    parameters = ['argc', 'argv']
    stmts = [Assignment(Target([name]), Parameter(i)) for [i, name] in enumerate(parameters)] + stmts

    block = Block(stmts)

    [graph, ctx, block] = graphulate(block)

    [graph, first, last] = block.gen_blocks(graph, ctx)
    for b in walk_blocks(graph, first):
        ctx = ctx <- .current_block = b
        [graph, ctx] = b.gen_ssa(graph, ctx)

    graph = propagate_phis(ctx, graph, first)

    # Simplify any expressions. This is a basic local optimization pass.
    #for b in walk_blocks(graph, first):
    #    [graph, ctx] = b.simplify(graph, ctx)

    # Almost done: for each expression node, generate machine instructions that will be
    # passed to the register allocator.
    [basic_blocks, node_map] = generate_lir(graph, first)

    return {'_main': compiler.compile(parameters, basic_blocks, node_map)}
