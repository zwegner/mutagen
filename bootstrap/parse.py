#!/usr/bin/env python3
import os
import sys

import ply.yacc as yacc

import lexer
import mg_builtins
from syntax import *

from lexer import tokens

root_dir = os.path.dirname(sys.path[0])
stdlib_dir = '%s/stdlib' % root_dir

start = 'stmt_list_or_empty'

precedence = [
    ['right', 'EQUALS'],
    ['left', 'OR'],
    ['left', 'AND'],
    ['left', 'NOT'],
    ['left', 'EQUALS_EQUALS', 'NOT_EQUALS', 'GREATER', 'GREATER_EQUALS',
        'LESS', 'LESS_EQUALS', 'IN'],
    ['left', 'BIT_OR'],
    ['left', 'BIT_XOR'],
    ['left', 'BIT_AND'],
    ['left', 'SHIFT_RIGHT', 'SHIFT_LEFT'],
    ['left', 'PLUS', 'MINUS'],
    ['left', 'STAR', 'FLOORDIV', 'MODULO'],
    ['left', 'UNARY_MINUS', 'INVERSE'],
    ['right', 'STAR_STAR'],
    ['left', 'LBRACKET', 'LPAREN', 'LBRACE'],
    ['left', 'PERIOD'],
]

def get_info(p, idx):
    return Info(filename, p.lineno(idx))

def p_error(p):
    # WHY IS THIS HAPPENING
    l = p.lexer
    if hasattr(l, 'lexer'):
        l = l.lexer
    print('%s(%i): %s' % (filename, l.lineno, p), file=sys.stderr)
    sys.exit(1)

def p_stmt_list_or_empty(p):
    """ stmt_list_or_empty : stmt_list
                           |
    """
    p[0] = p[1] if len(p) == 2 else []

def p_stmt_list(p):
    """ stmt_list : stmt """
    p[0] = []
    if p[1] is not None:
        p[0].append(p[1])

def p_stmt_list_2(p):
    """ stmt_list : stmt_list stmt """
    p[0] = p[1]
    if p[2] is not None:
        p[0].append(p[2])

def p_pass(p):
    """ pass : PASS """
    p[0] = None

def p_block(p):
    """ block : COLON delims INDENT stmt_list DEDENT
              | LBRACE stmt_list RBRACE
    """
    if len(p) == 4:
        p[0] = Block(p[2], info=get_info(p, 1))
    else:
        p[0] = Block(p[4], info=get_info(p, 1))

def p_block_single_stmt(p):
    """ block : COLON simple_stmt delim """
    stmts = []
    if p[2] is not None:
        stmts.append(p[2])
    p[0] = Block(stmts, info=get_info(p, 1))

def p_delim(p):
    """ delim : NEWLINE
              | SEMICOLON
    """
    p[0] = None

def p_delims(p):
    """ delims : delim
               | delims delim
    """
    p[0] = None

def p_ident_list(p):
    """ ident_list : IDENTIFIER
                   | ident_list COMMA IDENTIFIER
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_import(p):
    """ import : IMPORT IDENTIFIER
               | FROM IDENTIFIER IMPORT ident_list
               | FROM IDENTIFIER IMPORT STAR
               | IMPORT IDENTIFIER FROM STRING
    """
    names = path = None
    if len(p) == 5:
        if p[1] == 'from':
            names = p[4] if p[4] != '*' else []
        else:
            path = p[4]
    imp = Import([], p[2], names, path, False, info=get_info(p, 1))
    all_imports.append(imp)
    target = names or p[2]
    p[0] = Scope(imp)

def p_simple_stmt(p):
    """ simple_stmt : expr
                    | pass
                    | assn
                    | break
                    | continue
                    | return
                    | yield
                    | import
                    | assert
    """
    p[0] = p[1]

def p_stmt(p):
    """ stmt : simple_stmt delim
             | if_stmt
             | for_stmt
             | while_stmt
             | def_stmt
             | class_stmt
    """
    p[0] = p[1]

def p_stmt_2(p):
    """ stmt : stmt delim """
    p[0] = p[1]

def p_expr_list(p):
    """ expr_list : expr
                  | expr_list COMMA expr
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_expr(p):
    """ expr : list
             | list_comp
             | dict
             | dict_comp
             | set
             | lambda
             | ident
             | string
             | integer
             | boolean
             | unop
             | binop
             | getattr
             | getitem
             | call
             | none
    """
    p[0] = p[1]

def p_paren_expr(p):
    """ expr : LPAREN expr RPAREN """
    p[0] = p[2]

def p_ident(p):
    """ ident : IDENTIFIER """
    p[0] = Identifier(p[1], info=get_info(p, 1))

def p_string(p):
    """ string : STRING """
    p[0] = String(p[1], info=get_info(p, 1))

def p_integer(p):
    """ integer : INTEGER """
    p[0] = Integer(p[1], info=get_info(p, 1))

def p_bool(p):
    """ boolean : TRUE
                | FALSE
    """
    val = {'True': True, 'False': False}[p[1]]
    p[0] = Boolean(val, info=get_info(p, 1))

def p_list(p):
    """ list : LBRACKET expr_list RBRACKET
             | LBRACKET expr_list COMMA RBRACKET
             | LBRACKET RBRACKET
    """
    if len(p) == 3:
        p[0] = List([], info=get_info(p, 1))
    else:
        p[0] = List(p[2], info=get_info(p, 1))

def p_dict_items(p):
    """ dict_items : expr COLON expr
                   | dict_items COMMA expr COLON expr
    """
    if len(p) == 4:
        p[0] = [[p[1], p[3]]]
    else:
        p[0] = p[1] + [[p[3], p[5]]]

def p_comprehension(p):
    """ comp_iter : FOR for_assn IN expr """
    p[0] = CompIter(p[2], p[4])

def p_comp_list(p):
    """ comp_list : comp_iter
                  | comp_list comp_iter
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_list_comp(p):
    """ list_comp : LBRACKET expr comp_list RBRACKET """
    p[0] = Scope(ListComprehension(p[2], p[3]))

def p_dict_comp(p):
    """ dict_comp : LBRACE expr COLON expr comp_list RBRACE """
    p[0] = Scope(DictComprehension(p[2], p[4], p[5]))

def p_dict(p):
    """ dict : LBRACE RBRACE
             | LBRACE dict_items RBRACE
             | LBRACE dict_items COMMA RBRACE
    """
    if len(p) == 3:
        p[0] = Dict({}, info=get_info(p, 1))
    else:
        p[0] = Dict(dict(p[2]), info=get_info(p, 1))

def p_set(p):
    """ set : LBRACE expr_list RBRACE
            | LBRACE expr_list COMMA RBRACE
    """
    p[0] = Call(Identifier('set', info=get_info(p, 1)),
            [List(p[2], info=get_info(p, 1))])

def p_none(p):
    """ none : NONE """
    p[0] = None_(info=get_info(p, 1))

def p_unary_op(p):
    """ unop : NOT expr
             | MINUS expr %prec UNARY_MINUS
             | INVERSE expr
    """
    p[0] = UnaryOp(p[1], p[2])

def p_binary_op(p):
    """ binop : expr EQUALS_EQUALS expr
              | expr NOT_EQUALS expr
              | expr GREATER expr
              | expr GREATER_EQUALS expr
              | expr LESS expr
              | expr LESS_EQUALS expr
              | expr SHIFT_RIGHT expr
              | expr SHIFT_LEFT expr
              | expr PLUS expr
              | expr MINUS expr
              | expr STAR expr
              | expr STAR_STAR expr
              | expr FLOORDIV expr
              | expr MODULO expr
              | expr AND expr
              | expr OR expr
              | expr BIT_AND expr
              | expr BIT_OR expr
              | expr BIT_XOR expr
    """
    p[0] = BinaryOp(p[2], p[1], p[3])

# XXX since 'x in y' calls y.contains(x), i.e. the lhs and rhs are reversed from
# other binary operators, we have a special form here that swaps them.
def p_binary_op_in(p):
    """ binop : expr IN expr """
    p[0] = BinaryOp('in', p[3], p[1])

def p_binary_op_not_in(p):
    """ binop : expr NOT IN expr %prec IN """
    p[0] = UnaryOp('not', BinaryOp('in', p[4], p[1]))

def p_getattr(p):
    """ getattr : expr PERIOD IDENTIFIER """
    p[0] = GetAttr(p[1], p[3])

def p_getitem(p):
    """ getitem : expr LBRACKET expr RBRACKET """
    p[0] = GetItem(p[1], p[3])

def p_vararg(p):
    """ vararg : STAR expr """
    p[0] = VarArg(p[2])

def p_arg_list(p):
    """ arg_list : expr
                 | vararg
                 | arg_list COMMA expr
                 | arg_list COMMA vararg
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_call(p):
    """ call : expr LPAREN arg_list RPAREN
             | expr LPAREN RPAREN
    """
    if len(p) == 4:
        p[0] = Call(p[1], [])
    else:
        p[0] = Call(p[1], p[3])

def p_assert(p):
    """ assert : ASSERT expr """
    p[0] = Assert(p[2])

def p_assignment(p):
    """ assn : expr EQUALS expr """
    # Please god, die in a fire
    def deconstruct_lhs(lhs):
        if isinstance(lhs, Identifier):
            return lhs.name
        elif isinstance(lhs, List):
            return [deconstruct_lhs(i) for i in lhs]
        lhs.error('invalid lhs for assignment')
    p[0] = Assignment(Target(deconstruct_lhs(p[1]), info=get_info(p, 1)), p[3])

def p_break(p):
    """ break : BREAK """
    p[0] = Break(info=get_info(p, 1))

def p_continue(p):
    """ continue : CONTINUE """
    p[0] = Continue(info=get_info(p, 1))

def p_return(p):
    """ return : RETURN expr
               | RETURN
    """
    if len(p) == 3:
        p[0] = Return(p[2])
    else:
        p[0] = Return(None)

def p_yield(p):
    """ yield : YIELD expr """
    p[0] = Yield(p[2])

def p_if(p):
    """ if_stmt : IF expr block """
    p[0] = IfElse(p[2], p[3], Block([], info=get_info(p, 1)))

def p_ifelse(p):
    """ if_stmt : IF expr block elif_stmt
                | IF expr block else_stmt
    """
    p[0] = IfElse(p[2], p[3], p[4])

def p_elif(p):
    """ elif_stmt : ELIF expr block """
    stmts = [IfElse(p[2], p[3], Block([], info=get_info(p, 1)))]
    p[0] = Block(stmts, info=get_info(p, 1))

def p_elif_2(p):
    """ elif_stmt : ELIF expr block elif_stmt
                  | ELIF expr block else_stmt
    """
    stmts = [IfElse(p[2], p[3], p[4])]
    p[0] = Block(stmts, info=get_info(p, 1))

def p_else(p):
    """ else_stmt : ELSE block """
    p[0] = p[2]

# For loop assignment grammar. This is not used for regular assignments, since
# that needs to differentiate regular expressions on their own line, and that
# would need a parser with more lookahead, since we just see [<identifier>.
# We don't use <expr> as the assignment does, since 'expr in expr' is a
# containment test, but that also could match 'for expr in expr'... icky!
def p_for_assn_list(p):
    """ for_assn_list : for_assn_base
                      | for_assn_list COMMA for_assn_base
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_for_assn(p):
    """ for_assn_base : IDENTIFIER
                      | LBRACKET for_assn_list RBRACKET
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[2]

def p_for_assn_target(p):
    """ for_assn : for_assn_base """
    p[0] = Target(p[1], info=get_info(p, 1))

def p_for(p):
    """ for_stmt : FOR for_assn IN expr block """
    p[0] = For(p[2], p[4], p[5])

def p_while(p):
    """ while_stmt : WHILE expr block """
    p[0] = While(p[2], p[3])

def p_typespec(p):
    """ typespec : expr """
    p[0] = p[1]

def p_param(p):
    """ param : IDENTIFIER
              | IDENTIFIER COLON typespec
    """
    if len(p) == 2:
        p[0] = [p[1], None]
    else:
        p[0] = [p[1], p[3]]

def p_param_list(p):
    """ param_list : param
                   | param_list COMMA param
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_params(p):
    """ params : LPAREN param_list RPAREN
               | LPAREN RPAREN
               |
    """
    if len(p) == 4:
        params = p[2]
    else:
        params = []
    p[0] = Params(params, None, info=get_info(p, 0))

def p_params_star(p):
    """ params : LPAREN param_list COMMA STAR IDENTIFIER RPAREN
               | LPAREN STAR IDENTIFIER RPAREN
    """
    if len(p) == 5:
        p[0] = Params([], p[3], info=get_info(p, 1))
    else:
        p[0] = Params(p[2], p[5], info=get_info(p, 1))

def p_return_type(p):
    """ return_type : RARROW expr
                    |
    """
    if len(p) == 3:
        p[0] = p[2]
    else:
        p[0] = None

def p_decorator_list(p):
    """ decorator_list : AT expr NEWLINE
                       | decorator_list AT expr NEWLINE
    """
    if len(p) == 4:
        p[0] = [p[2]]
    else:
        p[0] = p[1] + [p[3]]

def p_decorators(p):
    """ decorators :
                   | decorator_list
    """
    if len(p) == 1:
        p[0] = []
    else:
        p[0] = p[1]

def p_def(p):
    """ def_stmt : decorators DEF IDENTIFIER params return_type block """
    fn = Scope(Function(p[3], p[4], p[5], p[6], info=get_info(p, 2)))
    for dec in p[1]:
        fn = Call(dec, [fn])
    p[0] = Assignment(Target(p[3], info=get_info(p, 3)), fn)

def p_lambda(p):
    """ lambda : LAMBDA params return_type block """
    p[0] = Scope(Function('lambda', p[2], p[3], p[4], info=get_info(p, 1)))

def p_class(p):
    """ class_stmt : CLASS IDENTIFIER params block """
    p[0] = Assignment(Target(p[2], info=get_info(p, 2)),
            Scope(Class(p[2], p[3], p[4], info=get_info(p, 1))))

parser = yacc.yacc(write_tables=0, debug=0)

all_imports = None
module_cache = {}
current_ctx = None

def parse(path, import_builtins=True, ctx=None):
    global all_imports, current_ctx, filename
    # Check if we've parsed this before. We do a check for recursive imports here too.
    if path in module_cache:
        if module_cache[path] is None:
            raise Exception('recursive import detected for %s' % path)
        return module_cache[path]
    module_cache[path] = None

    # Parse the file
    all_imports = []
    current_ctx = ctx
    filename = path
    dirname = os.path.dirname(path)
    if not dirname:
        dirname = '.'
    with open(path) as f:
        block = parser.parse(input=f.read(), lexer=lexer.get_lexer())

    # Do some post-processing, starting with adding builtins
    if import_builtins:
        builtins_path = '%s/__builtins__.mg' % stdlib_dir
        imp = Import([], 'builtins', [], builtins_path, True, info=builtin_info)
        all_imports.append(imp)
        block = [Scope(imp)] + block

    new_block = []

    for k, v in mg_builtins.builtins.items():
        new_block.append(Assignment(Target(k, info=builtin_info), v))

    # Recursively parse imports. Be sure to copy the all_imports since
    # we'll be clearing and modifying it in each child parsing pass.
    for imp in all_imports[:]:
        # Explicit path: use that
        if imp.path:
            import_paths = [imp.path, '%s/%s' % (dirname, imp.path)]
        else:
            import_paths = ['%s/%s.mg' % (cd, imp.name)
                for cd in [dirname, stdlib_dir]]
        # Normal import: find the file first in
        # the current directory, then stdlib
        for import_path in import_paths:
            if os.path.isfile(import_path):
                break
        else:
            print('checking paths: %s' % import_paths, file=sys.stderr)
            raise Exception('could not find import in path: %s' % imp.name)

        imp.parent_ctx = Context(imp.name, None, ctx)
        imp.stmts = parse(import_path, import_builtins=not imp.is_builtins,
                ctx=imp.parent_ctx)

    new_block.extend(block)

    # Be sure and return a duplicate of the list...
    module_cache[path] = new_block[:]
    return new_block

def interpret(path):
    ctx = Context('__main__', None, None)
    block = parse(path, ctx=ctx)
    preprocess_program(ctx, block)
    try:
        for expr in block:
            expr.eval(ctx)
    except ProgramError as e:
        if e.stack_trace:
            for line in e.stack_trace:
                print(line, file=sys.stderr)
        print(e.msg, file=sys.stderr)
        sys.exit(1)

interpret(sys.argv[1])
