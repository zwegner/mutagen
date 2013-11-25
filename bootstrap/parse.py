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

start = 'stmt_list'

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
    ['left', 'PLUS', 'MINUS'],
    ['left', 'STAR'],
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

def p_import(p):
    """ import : IMPORT IDENTIFIER
               | FROM IDENTIFIER IMPORT STAR
    """
    if len(p) == 3:
        p[0] = Import(p[2], None, None, False, info=get_info(p, 1))
    else:
        p[0] = Import(p[2], [], None, False, info=get_info(p, 1))

def p_import_from(p):
    """ import : IMPORT IDENTIFIER FROM STRING """
    p[0] = Import(p[2], None, p[4], False, info=get_info(p, 1))

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
             | union_stmt
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
             | dict
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

def p_dict_list(p):
    """ dict_list : expr COLON expr
                  | dict_list COMMA expr COLON expr
    """
    if len(p) == 4:
        p[0] = [[p[1], p[3]]]
    else:
        p[0] = p[1] + [[p[3], p[5]]]

def p_dict(p):
    """ dict : LBRACE RBRACE
             | LBRACE dict_list RBRACE
             | LBRACE dict_list COMMA RBRACE
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
    """ unop : NOT expr """
    p[0] = UnaryOp(p[1], p[2])

def p_binary_op(p):
    """ binop : expr EQUALS_EQUALS expr
              | expr NOT_EQUALS expr
              | expr GREATER expr
              | expr GREATER_EQUALS expr
              | expr LESS expr
              | expr LESS_EQUALS expr
              | expr MINUS expr
              | expr PLUS expr
              | expr STAR expr
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
    def deconstruct_lhs(lhs):
        if isinstance(lhs, Identifier):
            return lhs.name
        elif isinstance(lhs, List):
            return [deconstruct_lhs(i) for i in lhs]
        lhs.error('invalid lhs for assignment')
    p[0] = Assignment(deconstruct_lhs(p[1]), p[3])

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
    """ for_assn_list : for_assn
                      | for_assn_list COMMA for_assn
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_for_assn(p):
    """ for_assn : IDENTIFIER
                 | LBRACKET for_assn_list RBRACKET
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[2]

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

def p_def(p):
    """ def_stmt : DEF IDENTIFIER params block """
    p[0] = Assignment(p[2], Function(current_ctx, p[2], p[3], p[4], info=get_info(p, 1)))

def p_lambda(p):
    """ lambda : LAMBDA params block """
    p[0] = Function(current_ctx, 'lambda', p[2], p[3], info=get_info(p, 1))

def p_class(p):
    """ class_stmt : CLASS IDENTIFIER params block """
    p[0] = Assignment(p[2], Class(current_ctx, p[2], p[3], p[4], info=get_info(p, 1)))

# Union parameters have custom parse rules so that they can support inline
# class definitions.
def p_union_param(p):
    """ union_param : IDENTIFIER
                    | IDENTIFIER COLON typespec
                    | IDENTIFIER COLON CLASS params
    """
    if len(p) == 5:
        # For inline classes, pass the list of parameters wrapped in an object.
        # The class will be created inside the union.
        p[0] = [p[1], UnionInlineClass(p[4])]
    elif len(p) == 2:
        p[0] = [p[1], None]
    else:
        p[0] = [p[1], p[3]]

def p_union_param_list(p):
    """ union_param_list : union_param
                         | union_param_list COMMA union_param
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_union(p):
    """ union_stmt : UNION IDENTIFIER LPAREN union_param_list RPAREN block """
    p[0] = Assignment(p[2], Union(current_ctx, p[2], Params(p[4], None,
        info=get_info(p, 1)), p[6], info=get_info(p, 1)))

parser = yacc.yacc(write_tables=0, debug=0)

module_cache = {}
current_ctx = None

def parse(path, import_builtins=True, ctx=None):
    global current_ctx, filename
    # Check if we've parsed this before. We do a check for recursive imports here too.
    if path in module_cache:
        if module_cache[path] is None:
            raise Exception('recursive import detected for %s' % path)
        return module_cache[path]
    module_cache[path] = None

    # Parse the file
    current_ctx = ctx
    filename = path
    dirname = os.path.dirname(path)
    if not dirname:
        dirname = '.'
    with open(path) as f:
        block = parser.parse(input=f.read(), lexer=lexer.get_lexer())

    # Do some post-processing, starting with adding builtins
    if import_builtins:
        path = '%s/__builtins__.mg' % stdlib_dir
        block = [Import('builtins', [], path, True,
            info=builtin_info)] + block

    new_block = []

    for k, v in mg_builtins.builtins.items():
        new_block.append(Assignment(k, v))

    # Recursively parse imports
    for expr in block:
        if isinstance(expr, Import):
            # Explicit path: use that
            if expr.path:
                path = expr.path
            else:
                # Normal import: find the file first in
                # the current directory, then stdlib
                for cd in [dirname, stdlib_dir]:
                    path = '%s/%s.mg' % (cd, expr.name)
                    if os.path.isfile(path):
                        break
                else:
                    raise Exception('could not find import in path: %s' % expr.name)

            module_ctx = Context(expr.name, None, ctx)
            stmts = parse(path, import_builtins=not expr.is_builtins,
                    ctx=module_ctx)
            expr.ctx = module_ctx
            expr.stmts = stmts
    new_block.extend(block)

    # Be sure and return a duplicate of the list...
    module_cache[path] = new_block[:]
    return new_block

def interpret(path):
    ctx = Context('__main__', None, None)
    block = parse(path, ctx=ctx)
    block = ctx.initialize(block)
    for expr in block:
        expr.eval(ctx)

interpret(sys.argv[1])
