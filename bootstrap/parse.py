#!/usr/bin/env python3
import os
import sys

import ply.yacc as yacc

import lexer
import syntax
import mg_builtins

from lexer import tokens

start = 'block'

precedence = [
    ['right', 'EQUALS'],
    ['left', 'LBRACKET', 'LPAREN', 'PERIOD'],
]

def p_error(p):
    print('%s(%i): %s' % (syntax.filename, p.lexer.lexer.lineno, p))
    raise ParseError()

class ParseError(Exception):
    pass

def p_block_1(p):
    """ block : stmt """
    p[0] = []
    if p[1] is not None:
        p[0] += [p[1]]

def p_block_3(p):
    """ block : block stmt """
    p[0] = p[1]
    if p[2] is not None:
        p[0] += [p[2]]

def p_statement(p):
    """ stmt : expr NEWLINE
             | expr SEMICOLON
    """
    p[0] = p[1]

def p_statement_2(p):
    """ stmt : NEWLINE
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
    """ list : LBRACKET expr_list RBRACKET """
    p[0] = syntax.List(p[2])

def p_getattr(p):
    """ expr : expr PERIOD IDENTIFIER """
    p[0] = syntax.GetAttr(p[1], syntax.String(p[3]))

def p_def(p):
    """ def : list LBRACE block RBRACE """
    p[0] = syntax.Function(p[1], p[3])

def p_call(p):
    """ expr : expr LPAREN expr_list RPAREN """
    p[0] = syntax.Call(p[1], p[3])

def p_assignment(p):
    """ expr : IDENTIFIER EQUALS expr """
    p[0] = syntax.Assignment(p[1], p[3])

p = yacc.yacc()

def parse(path):
    dirname = os.path.dirname(path)
    if not dirname:
        dirname = '.'
    with open(path) as f:
        block = p.parse(input=f.read(), lexer=lexer.get_lexer())
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

    ctx = syntax.Context(None)
    for k, v in mg_builtins.builtins.items():
        ctx.store(k, v)

    for expr in block:
        expr.eval(ctx)

interpret(sys.argv[1])
