#!/usr/bin/python
import os
import sys

import ply.yacc as yacc

import lexer
import syntax
import mg_builtins

from lexer import tokens

start = 'block'

def p_error(p):
    raise ParseError()

class ParseError(Exception):
    pass

def p_newline(p):
    """ newline : NEWLINE
                | newline NEWLINE
    """
    pass

def p_block(p):
    """ block : NEWLINE
              | expr
              | block newline expr
    """
    print(len(p))
    if len(p) == 2:
        return [p[1]]
    else:
        return p[1] + [p[3]]

def p_import(p):
    """ expr : IMPORT IDENTIFIER """
    return syntax.Import(p[2].value)

def p_expr_list(p):
    """ expr_list : expr
                  | expr_list COMMA expr
    """
    if len(p) == 2:
        return [p[1]]
    else:
        return p[1] + [p[3]]

def p_expr(p):
    """ expr : LPAREN expr RPAREN
             | list
             | def
    """
    return p

def p_ident(p):
    """ expr : IDENTIFIER """
    return syntax.Identifier(p[1].value)

def p_string(p):
    """ expr : STRING """
    return syntax.String(p[1].value)

def p_list(p):
    """ list : LBRACKET expr_list RBRACKET """
    return syntax.List(p[2])

def p_def(p):
    """ def : list LBRACE block RBRACE """
    return syntax.Function(p[1], p[3])

def p_call(p):
    """ expr : expr LPAREN expr_list RPAREN """
    return syntax.Call(p[1], p[3])

def p_assignment(p):
    """ expr : IDENTIFIER EQUALS expr """
    return syntax.Assignment(p[1].value, p[3])

def parse(path):
    dirname = os.path.dirname(path)
    if not dirname:
        dirname = '.'
    with open(path) as f:
        p = yacc.yacc()
        block = p.parse(input=f.read(), lexer=lexer.get_lexer(), debug=1)
        print(block)
        p.expect('EOF')
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
