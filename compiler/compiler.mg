import dumb_regalloc
import lexer
import libgraph
# XXX only doing this because of weird importing behavior, to get the 'from syntax import *' goodies
from parse import *

# XXX work around weird importing behavior
asm = dumb_regalloc.asm
regalloc = dumb_regalloc.regalloc
lir = dumb_regalloc.lir

################################################################################
## Utilities ###################################################################
################################################################################

def block_name(block_id):
    return '$block{}'.format(block_id)

def hacky_type_check(obj, type_name):
    return type(obj).__name__ == 'G' + type_name

# Walk the CFG in preorder
def walk_blocks(graph, first):
    work = [first]
    seen = {first}
    while work:
        [block, work] = [work[0], work[1:]]
        yield block
        for succ in block.succs_gen(graph):
            if succ not in seen:
                work = work + [succ]
                seen = seen | {succ}

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

class SSAContext(new_class_dict: dict, current_block=None, statements=[], seen_set: set=set()):
    def load(self, graph, name):
        # This test assumes that exit_states is being constructed with a forward pass through the tree
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

################################################################################
## First pass: create a DAG representation #####################################
################################################################################

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
            attrs = attrs <- [attr] = lambda (self, graph): graph.get(self, attr) if attr in graph.edge_values[self] else None
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
    params = Parameters(names, types, params.var_params, kwparams, params.kw_var_params)

    return hacky_class_from_base_and_new_attrs('G' + cls.__name__, cls, attrs, params)

@fixed_point
def add_to_graph(add_to_graph, graph, ctx, node):
    new_class = ctx.get_graph_class(type(node))

    # XXX ughhhh... pick out all the non-node arguments to pass to the create_node call
    args = []
    for param in type(node).__params__.names:
        if not any([attr == param for [attr, edge_type] in new_class._node_edges]):
            args = args + [getattr(node, param)]

    [graph, new_node] = ctx.create_node(graph, new_class, *args)

    # Recurse through all of the child nodes
    for [attr, edge_type] in new_class._node_edges:
        if edge_type == '*':
            for item in getattr(node, attr):
                [graph, value] = add_to_graph(graph, ctx, item)
                graph = graph.append_to_edge(new_node, attr, value)
        elif edge_type == '#':
            for [key, value] in getattr(node, attr):
                #[graph, key] = add_to_graph(graph, ctx, key)
                [graph, value] = add_to_graph(graph, ctx, value)
                graph = graph.set_edge_key(new_node, key, value)
        else:
            value = getattr(node, attr)
            if edge_type != '?' or value != None:
                [graph, value] = add_to_graph(graph, ctx, value)
                graph = graph.set_edge(new_node, attr, value)
    return [graph, new_node]

# Take the input AST and create a directed graph out of it. We do some weird metaprogramming
# to accomplish this, which should go away once we have a better language...
def generate_graph(block):
    # Ugh
    new_class_dict = {cls.__name__: create_graph_class(cls) for cls in [Phi, BinaryOp, Integer, Identifier,
        Parameter, Assignment, BasicBlock, Block, While, Return, Call]}
    ctx = SSAContext(new_class_dict)

    graph = libgraph.DirectedGraph()
    [graph, block] = add_to_graph(graph, ctx, block)

    return [graph, ctx, block]

################################################################################
## Second pass: split up code into basic blocks ################################
################################################################################

def link_blocks(graph, pred, succ):
    graph = pred.succs_append(graph, succ)
    graph = succ.preds_append(graph, pred)
    return graph

@fixed_point
def gen_blocks(gen_blocks, graph, ctx, node):
    # XXX pattern matching
    if hacky_type_check(node, 'Block'):
        [graph, prelude] = ctx.create_node(graph, 'BasicBlock')
        curr_block = prelude
        # XXX this is insufficient, in that we're only looking through the top-level children for
        # nodes that need to be split up into basic blocks. Or maybe after we do syntax desugaring
        # etc., this will be enough?
        for stmt in node.stmts_gen(graph):
            # Try to generate blocks for the child, if this doesn't work, it's just a normal
            # node
            result = gen_blocks(graph, ctx, stmt)
            if result:
                [graph, first, last] = result
                graph = link_blocks(graph, curr_block, first)
                [graph, curr_block] = ctx.create_node(graph, 'BasicBlock')
                graph = link_blocks(graph, last, curr_block)
            else:
                graph = curr_block.stmts_append(graph, stmt)

        graph = graph.delete_node(node)
        return [graph, prelude, curr_block]
    elif hacky_type_check(node, 'While'):
        # Create blocks
        [graph, test_block] = ctx.create_node(graph, 'BasicBlock')
        test = node.test(graph)
        if test != None:
            graph = graph.set_edge_list(test_block, 'stmts', [test])
            graph = graph.set_edge(test_block, 'test', test)
        [graph, first, last] = gen_blocks(graph, ctx, node.block(graph))
        [graph, exit_block] = ctx.create_node(graph, 'BasicBlock')

        # Link them up
        graph = link_blocks(graph, test_block, first)
        graph = link_blocks(graph, test_block, exit_block)
        graph = link_blocks(graph, last, test_block)

        graph = graph.delete_node(node)
        return [graph, test_block, exit_block]
    elif hacky_type_check(node, 'IfElse'):
        # Create blocks
        [graph, test_block] = ctx.create_node(graph, 'BasicBlock')
        test = node.test(graph)
        graph = graph.set_edge_list(test_block, 'stmts', [test])
        graph = graph.set_edge(test_block, 'test', test)
        [graph, if_first, if_last] = gen_blocks(graph, ctx, node.if_block(graph))
        [graph, else_first, else_last] = gen_blocks(graph, ctx, node.else_block(graph))
        [graph, exit_block] = ctx.create_node(graph, 'BasicBlock')

        # Link them up
        graph = link_blocks(graph, test_block, if_first)
        graph = link_blocks(graph, test_block, else_first)
        graph = link_blocks(graph, if_last, exit_block)
        graph = link_blocks(graph, else_last, exit_block)

        graph = graph.delete_node(node)
        return [graph, test_block, exit_block]
    return None

def generate_blocks(graph, ctx, block):
    [graph, first, last] = gen_blocks(graph, ctx, block)
    return [graph, first]

################################################################################
## Third pass: convert graph to SSA ############################################
################################################################################

@fixed_point
def propagate_phi(propagate_phi, graph, ctx, block, phi_name):
    for pred in block.preds_gen(graph):
        if phi_name not in pred.exit_states(graph):
            # Add a duplicate phi to the block's phi list and its exit states
            [graph, phi] = ctx.create_node(graph, 'Phi', phi_name)
            graph = graph.append_to_edge(pred, 'phis', phi)
            graph = graph.set_edge_key(pred, 'exit_states', phi_name, phi)

            # Now recursively propagate the phi backward through this block's predecessors
            graph = propagate_phi(graph, ctx, pred, phi_name)

    return graph

def fix_phis(graph, ctx, first_block):
    # Iterate through the graph and add phis to blocks whose successors load from a given
    # variable before they are assigned, which means
    for block in walk_blocks(graph, first_block):
        for phi in block.phis_gen(graph):
            graph = propagate_phi(graph, ctx, block, phi.name)

    return graph

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
        if child != None and child.node_id not in ctx.seen_set:
            ctx = ctx <- .statements += [child], .seen_set |= {child.node_id}

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

def generate_ssa(graph, ctx, first_block):
    for block in walk_blocks(graph, first_block):
        ctx = ctx <- .current_block = block

        new_stmts = []
        for stmt in block.stmts_gen(graph):
            ctx = ctx <- .statements = []

            [graph, ctx, new_stmt] = rec_gen_ssa(stmt, graph, ctx)
            new_stmts = new_stmts + ctx.statements
            if new_stmt != None:
                new_stmts = new_stmts + [new_stmt]

        graph = graph.set_edge_list(block, 'stmts', new_stmts)

    graph = fix_phis(graph, ctx, first_block)

    return [graph, ctx]

################################################################################
## Fourth pass: try to simplify any nodes we can ###############################
################################################################################

def simplify_node(graph, ctx, node):
    if hacky_type_check(node, 'BinaryOp'):
        if (hacky_type_check(node.lhs(graph), 'Integer') and
                hacky_type_check(node.rhs(graph), 'Integer')):
            [lhs, rhs] = [node.lhs(graph).value, node.rhs(graph).value]
            # Urgh
            if node.op == '+':
                result = lhs + rhs
            elif node.op == '-':
                result = lhs - rhs
            elif node.op == '&':
                result = lhs & rhs
            elif node.op == '|':
                result = lhs | rhs
            else:
                print(node.op)
                assert False
            [graph, new_node] = ctx.create_node(graph, 'Integer', result)
            graph = graph.replace_node(node, new_node)
            return graph
    return graph

DCE_CLASS_WHITELIST = ['BinaryOp', 'Integer']

def simplify_blocks(graph, ctx, first_block):
    # Simplify any expressions. This is a basic local optimization pass.
    for block in walk_blocks(graph, first_block):
        for stmt in block.stmts_gen(graph):
            graph = simplify_node(graph, ctx, stmt)

    # Dead code elimination
    while True:
        any_removed = False
        for block in walk_blocks(graph, first_block):
            new_stmts = []
            # Eliminate any non-side-effecting expressions (which should be everything but it's
            # not yet) that aren't referenced elsewhere (the only reference being this block).
            for stmt in block.stmts_gen(graph):
                if (any([hacky_type_check(stmt, cls) for cls in DCE_CLASS_WHITELIST]) and
                        len(graph.get_uses(stmt)) == 1):
                    any_removed = True
                else:
                    new_stmts = new_stmts + [stmt]
            graph = graph.set_edge_list(block, 'stmts', new_stmts)

            # Remove entries from exit_states if they aren't in any phis of successor blocks
            for [key, _] in block.exit_states(graph):
                if not any([phi.name == key for succ in block.succs_gen(graph)
                        for phi in succ.phis_gen(graph)]):
                    graph = graph.unset_edge_key(block, 'exit_states', key)
                    any_removed = True

        if not any_removed:
            break

    return graph

################################################################################
## Fifth pass: convert SSA to LIR ##############################################
################################################################################

BINOP_TABLE = {
    '+': ['__add__', lir.add64],
    '-': ['__sub__', lir.sub64],
    '&': ['__and__', lir.and64],
    '|': ['__or__',  lir.or64],
}

def gen_insts_for_node(graph, node):
    # XXX pattern matching
    if hacky_type_check(node, 'Phi'):
        return lir.phi_ref(node.name)
    elif hacky_type_check(node, 'BinaryOp'):
        fn = BINOP_TABLE[node.op][1]
        return fn(lir.NR(node.lhs(graph).node_id), lir.NR(node.rhs(graph).node_id))
    elif hacky_type_check(node, 'Parameter'):
        return lir.parameter(node.index)
    elif hacky_type_check(node, 'Integer'):
        return lir.mov64(node.value)
    elif hacky_type_check(node, 'Call'):
        # Stupid temporary hack
        fn = node.fn(graph).name
        if fn in {'test', 'test2', 'deref', 'atoi'}:
            fn = asm.ExternLabel('_' + fn)

        return lir.call(fn, *[lir.NR(arg.node_id) for arg in node.args_gen(graph)])
    elif hacky_type_check(node, 'Return'):
        return lir.ret(lir.NR(node.expr(graph).node_id))
    assert False

# Walk over the blocks and convert the SSA DAG to LIR (low-level intermediate representation).
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
            inst = gen_insts_for_node(graph, stmt)
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

################################################################################
## Parent functions ############################################################
################################################################################

# Run all of the compilation passes on a given function
def compile_function(parameters, block):
    [graph, ctx, block] = generate_graph(block)

    [graph, first_block] = generate_blocks(graph, ctx, block)

    [graph, ctx] = generate_ssa(graph, ctx, first_block)

    graph = simplify_blocks(graph, ctx, first_block)

    [basic_blocks, node_map] = generate_lir(graph, first_block)

    return dumb_regalloc.Function(parameters, basic_blocks, node_map)

def compile(stmts):
    # Add statements for parameter handling
    parameters = ['argc', 'argv']
    stmts = [Assignment(Target([name]), Parameter(i)) for [i, name] in enumerate(parameters)] + stmts

    block = Block(stmts)

    # XXX pull out functions

    return {'_main': compile_function(parameters, block)}

line = read_file('new.mg')
x = parser.parse(lexer.input(line))
dumb_regalloc.export_functions('elfout.o', compile(x))
