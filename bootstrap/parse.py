#!/usr/bin/env python3
import os
import sys

import ply.yacc as yacc

import lexer
import syntax
import mg_builtins

from lexer import tokens

root_dir = os.path.dirname(sys.path[0])
stdlib_dir = '%s/stdlib' % root_dir

start = 'stmt_list'

precedence = [
    ['right', 'EQUALS'],
    ['left', 'OR'],
    ['left', 'AND'],
    ['left', 'NOT'],
    ['left', 'EQUALS_EQUALS', 'NOT_EQUALS', 'GREATER', 'GREATER_EQUALS', 'LESS', 'LESS_EQUALS'],
    ['left', 'PLUS', 'MINUS'],
    ['left', 'LBRACKET', 'LPAREN', 'LBRACE'],
    ['left', 'PERIOD'],
]

def get_info(p, idx):
    return syntax.Info(syntax.filename, p.lineno(idx))

def p_error(p):
    # WHY IS THIS HAPPENING
    l = p.lexer
    if hasattr(l, 'lexer'):
        l = l.lexer
    print('%s(%i): %s' % (syntax.filename, l.lineno, p), file=sys.stderr)
    sys.exit(1)

def p_stmt_list(p):
    """ stmt_list : stmt """
    p[0] = [p[1]]

def p_stmt_list_2(p):
    """ stmt_list : stmt_list delim
                  | stmt_list stmt
    """
    p[0] = p[1]
    if p[2] is not None:
        p[0].append(p[2])

def p_pass(p):
    """ pass : PASS """
    p[0] = None

def p_block(p):
    """ block : COLON delim INDENT stmt_list DEDENT
              | LBRACE stmt_list RBRACE
    """
    if len(p) == 4:
        p[0] = p[2]
    else:
        p[0] = p[4]

def p_delim(p):
    """ delim : NEWLINE
              | SEMICOLON
              | delim NEWLINE
              | delim SEMICOLON
    """
    p[0] = None

def p_import(p):
    """ import : IMPORT IDENTIFIER
               | FROM IDENTIFIER IMPORT STAR
    """
    if len(p) == 3:
        p[0] = syntax.Import(p[2], None, None, False, info=get_info(p, 1))
    else:
        p[0] = syntax.Import(p[2], [], None, False, info=get_info(p, 1))

def p_import_from(p):
    """ import : IMPORT IDENTIFIER FROM STRING """
    p[0] = syntax.Import(p[2], None, p[4], False, info=get_info(p, 1))

def p_stmt(p):
    """ stmt : expr delim
             | pass
             | import
             | if_stmt
             | assn
             | for_stmt
             | while_stmt
             | def_stmt
             | class_stmt
    """
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
             | lambda
             | ident
             | string
             | integer
             | unop
             | binop
             | getattr
             | getitem
             | call
             | nil
    """
    p[0] = p[1]

def p_paren_expr(p):
    """ expr : LPAREN expr RPAREN """
    p[0] = p[2]

def p_ident(p):
    """ ident : IDENTIFIER """
    p[0] = syntax.Identifier(p[1], info=get_info(p, 1))

def p_string(p):
    """ string : STRING """
    p[0] = syntax.String(p[1], info=get_info(p, 1))

def p_integer(p):
    """ integer : INTEGER """
    p[0] = syntax.Integer(p[1], info=get_info(p, 1))

def p_list(p):
    """ list : LBRACKET expr_list RBRACKET
             | LBRACKET expr_list COMMA RBRACKET
             | LBRACKET RBRACKET
    """
    if len(p) == 3:
        p[0] = syntax.List([], info=get_info(p, 1))
    else:
        p[0] = syntax.List(p[2], info=get_info(p, 1))

def p_nil(p):
    """ nil : NIL """
    p[0] = syntax.Nil(info=get_info(p, 1))

def p_unary_op(p):
    """ unop : NOT expr """
    p[0] = syntax.UnaryOp(p[1], p[2])

def p_binary_op(p):
    """ binop : expr EQUALS_EQUALS expr
              | expr NOT_EQUALS expr
              | expr GREATER expr
              | expr GREATER_EQUALS expr
              | expr LESS expr
              | expr LESS_EQUALS expr
              | expr MINUS expr
              | expr PLUS expr
              | expr AND expr
              | expr OR expr
    """
    p[0] = syntax.BinaryOp(p[2], p[1], p[3])

def p_getattr(p):
    """ getattr : expr PERIOD IDENTIFIER """
    p[0] = syntax.GetAttr(p[1], p[3])

def p_getitem(p):
    """ getitem : expr LBRACKET expr RBRACKET """
    p[0] = syntax.GetItem(p[1], p[3])

def p_call(p):
    """ call : expr LPAREN expr_list RPAREN
             | expr LPAREN RPAREN
    """
    if len(p) == 4:
        p[0] = syntax.Call(p[1], [])
    else:
        p[0] = syntax.Call(p[1], p[3])

def p_assignment(p):
    """ assn : expr EQUALS expr """
    def deconstruct_lhs(lhs):
        if isinstance(lhs, syntax.Identifier):
            return lhs.name
        elif isinstance(lhs, syntax.List):
            return [deconstruct_lhs(i) for i in lhs]
        lhs.error('invalid lhs for assignment')
    p[0] = syntax.Assignment(deconstruct_lhs(p[1]), p[3])

def p_if(p):
    """ if_stmt : IF expr block """
    p[0] = syntax.IfElse(p[2], p[3], [])

def p_ifelse(p):
    """ if_stmt : IF expr block elif_stmt
                | IF expr block else_stmt
    """
    p[0] = syntax.IfElse(p[2], p[3], p[4])

def p_elif(p):
    """ elif_stmt : ELIF expr block """
    p[0] = [syntax.IfElse(p[2], p[3], [])]

def p_elif_2(p):
    """ elif_stmt : ELIF expr block elif_stmt
                  | ELIF expr block else_stmt
    """
    p[0] = [syntax.IfElse(p[2], p[3], p[4])]

def p_else(p):
    """ else_stmt : ELSE block """
    p[0] = p[2]

def p_for(p):
    """ for_stmt : FOR IDENTIFIER IN expr block """
    p[0] = syntax.For(p[2], p[4], p[5])

def p_while(p):
    """ while_stmt : WHILE expr block """
    p[0] = syntax.While(p[2], p[3])

def p_arg_list(p):
    """ arg_list : IDENTIFIER
                 | arg_list COMMA IDENTIFIER
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_args(p):
    """ args : LPAREN arg_list RPAREN
             | LPAREN RPAREN
    """
    if len(p) == 3:
        p[0] = []
    else:
        p[0] = p[2]

def p_def(p):
    """ def_stmt : DEF IDENTIFIER args block """
    p[0] = syntax.Assignment(p[2], syntax.Function(current_ctx, p[2], p[3], p[4], info=get_info(p, 1)))

def p_lambda(p):
    """ lambda : LAMBDA args block """
    p[0] = syntax.Function(current_ctx, 'lambda', p[2], p[3], info=get_info(p, 1))

def p_class(p):
    """ class_stmt : CLASS IDENTIFIER block """
    p[0] = syntax.Assignment(p[2], syntax.Class(current_ctx, p[2], p[3], info=get_info(p, 1)))

parser = yacc.yacc(write_tables=0, debug=0)

module_cache = {}
current_ctx = None

def parse(path, import_builtins=True, ctx=None):
    global current_ctx
    # Check if we've parsed this before. We do a check for recursive imports here too.
    if path in module_cache:
        if module_cache[path] is None:
            raise Exception('recursive import detected for %s' % path)
        return module_cache[path]
    module_cache[path] = None

    # Parse the file
    current_ctx = ctx
    syntax.filename = path
    dirname = os.path.dirname(path)
    if not dirname:
        dirname = '.'
    with open(path) as f:
        block = parser.parse(input=f.read(), lexer=lexer.get_lexer())

    # Do some post-processing, starting with adding builtins
    if import_builtins:
        path = '%s/__builtins__.mg' % stdlib_dir
        block = [syntax.Import('builtins', [], path, True,
            info=syntax.Info('__builtins__', 0))] + block

    new_block = []

    for k, v in mg_builtins.builtins.items():
        new_block.append(syntax.Assignment(k, v))

    # Recursively parse imports
    for expr in block:
        if isinstance(expr, syntax.Import):
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

            module_ctx = syntax.Context(expr.name, expr, None, ctx)
            stmts = parse(path, import_builtins=not expr.is_builtins,
                    ctx=module_ctx)
            expr.ctx = module_ctx
            expr.stmts = stmts
    new_block.extend(block)

    # Be sure and return a duplicate of the list...
    module_cache[path] = new_block[:]
    return new_block

def interpret(path):
    ctx = syntax.Context('__main__', None, None, None)
    block = parse(path, ctx=ctx)
    for expr in block:
        expr.eval(ctx)

interpret(sys.argv[1])
