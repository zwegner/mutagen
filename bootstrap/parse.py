#!/usr/bin/env python3
import os
import sys

import ply.yacc as yacc

import lexer
import syntax
import mg_builtins

from lexer import tokens

start = 'stmt_list'

precedence = [
    ['right', 'EQUALS'],
    ['left', 'NOT'],
    ['left', 'EQUALS_EQUALS'],
    ['left', 'PLUS'],
    ['left', 'LBRACKET', 'LPAREN', 'LBRACE'],
    ['left', 'PERIOD'],
]

def p_error(p):
    # WHY IS THIS HAPPENING
    l = p.lexer
    if hasattr(l, 'lexer'):
        l = l.lexer
    print('%s(%i): %s' % (syntax.filename, l.lineno, p))
    raise ParseError()

class ParseError(Exception):
    pass

def p_stmt_list(p):
    """ stmt_list : expr """
    p[0] = [p[1]]

def p_stmt_list_2(p):
    """ stmt_list : stmt_list expr """
    p[0] = p[1] + [p[2]]

def p_stmt_list_3(p):
    """ stmt_list : stmt_list delim """
    p[0] = p[1]

def p_pass(p):
    """ stmt_list : PASS """
    p[0] = []

def p_block(p):
    """ block : COLON delim INDENT stmt_list DEDENT """
    p[0] = p[4]

def p_delim(p):
    """ delim : NEWLINE
              | SEMICOLON
    """
    p[0] = None

def p_import(p):
    """ expr : IMPORT IDENTIFIER """
    p[0] = syntax.Import(p[2])

def p_expr_list(p):
    """ expr_list : expr
                  | expr_list COMMA expr
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_expr(p):
    """ expr : LPAREN expr RPAREN
             | list
             | def
    """
    if len(p) == 4:
        p[0] = p[2]
    else:
        p[0] = p[1]

def p_ident(p):
    """ expr : IDENTIFIER """
    p[0] = syntax.Identifier(p[1])

def p_string(p):
    """ expr : STRING """
    p[0] = syntax.String(p[1])

def p_integer(p):
    """ expr : INTEGER """
    p[0] = syntax.Integer(p[1])

def p_list(p):
    """ list : LBRACKET expr_list RBRACKET
             | LBRACKET RBRACKET
    """
    if len(p) == 3:
        p[0] = syntax.List([])
    else:
        p[0] = syntax.List(p[2])

def p_nil(p):
    """ expr : NIL """
    p[0] = syntax.Nil()

def p_unary_op(p):
    """ expr : NOT expr """
    p[0] = syntax.UnaryOp(p[1], p[2])

def p_binary_op(p):
    """ expr : expr EQUALS_EQUALS expr
             | expr PLUS expr
    """
    p[0] = syntax.BinaryOp(p[2], p[1], p[3])

def p_getattr(p):
    """ expr : expr PERIOD IDENTIFIER """
    p[0] = syntax.GetAttr(p[1], p[3])

def p_getitem(p):
    """ expr : expr LBRACKET expr RBRACKET """
    p[0] = syntax.GetItem(p[1], p[3])

def p_call(p):
    """ expr : expr LPAREN expr_list RPAREN """
    p[0] = syntax.Call(p[1], p[3])

def p_assignment(p):
    """ expr : IDENTIFIER EQUALS expr """
    p[0] = syntax.Assignment(p[1], p[3])

def p_if(p):
    """ expr : IF expr block """
    p[0] = syntax.IfElse(p[2], p[3], [])

def p_ifelse(p):
    """ expr : IF expr block elif_stmt
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

def p_arg_list(p):
    """ arg_list : IDENTIFIER
                 | arg_list COMMA IDENTIFIER
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_args(p):
    """ args : LPAREN arg_list RPAREN """
    if len(p) == 3:
        p[0] = []
    else:
        p[0] = p[2]

def p_def(p):
    """ def : DEF IDENTIFIER args block """
    p[0] = syntax.Assignment(p[2], syntax.Function(p[2], p[3], p[4]))

def p_lambda(p):
    """ def : LAMBDA args block """
    p[0] = syntax.Function('lambda', p[2], p[3])

parser = yacc.yacc()

def parse(path):
    dirname = os.path.dirname(path)
    if not dirname:
        dirname = '.'
    with open(path) as f:
        block = parser.parse(input=f.read(), lexer=lexer.get_lexer())
    # Recursively parse imports
    # XXX Currently doesn't check for infinite recursion, and 
    # imports everything into the same namespace
    new_block = []
    for expr in block:
        if isinstance(expr, syntax.Import):
            module = parse('%s/%s.mg' % (dirname, expr.name))
            new_block.extend(module)
        else:
            new_block.append(expr)

    return new_block

def interpret(path):
    block = parse(path)

    ctx = syntax.Context('<global>', None)
    for k, v in mg_builtins.builtins.items():
        ctx.store(k, v)

    for expr in block:
        expr.eval(ctx)

interpret(sys.argv[1])
