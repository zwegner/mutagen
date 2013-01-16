import sys

import lexer
import syntax
import mg_builtins

class ParseError(Exception):
    pass

class Parser:
    def __init__(self, input_file):
        self.lexer = lexer.Lexer(input_file)
        self.next_token()

    def accept(self, token):
        if self.token[0] == token:
            self.token_v = self.token[1]
            self.next_token()
            return True
        return False

    def expect(self, token):
        if not self.accept(token):
            raise ParseError()

    def parse_block(self):
        exprs = []
        while True:
            expr = self.parse_expr()
            if not expr:
                if self.accept('newline'):
                    continue
                break
            exprs.append(expr)
            if not self.accept('newline'):
                break
        return exprs

    def parse_list(self):
        exprs = []
        while True:
            expr = self.parse_expr()
            if not expr:
                break
            exprs.append(expr)
            if not self.accept('comma'):
                break
        return exprs

    def parse_expr(self, level=0):
        if level == -2:
            # Identifiers
            if self.accept('ident'):
                return syntax.Identifier(self.token_v)
            # String literals
            elif self.accept('string'):
                return syntax.String(self.token_v)
            # Parenthesitized exprs
            elif self.accept('lparen'):
                expr = self.parse_expr()
                self.expect('rparen')
                return expr
            elif self.accept('lbracket'):
                params = self.parse_list()
                self.expect('rbracket')
                # Function def
                if self.accept('lbrace'):
                    if any(not isinstance(p, syntax.Identifier) for p in params):
                        raise ParseError()
                    block = self.parse_block()
                    self.expect('rbrace')
                    return syntax.Function(params, block)
                return syntax.List(params)
            return None
        elif level == -1:
            expr = self.parse_expr(level - 1)
            if not expr:
                return None
            # Function call
            while self.accept('lparen'):
                args = self.parse_list()
                self.expect('rparen')
                expr = syntax.Call(expr, args)
            return expr
        elif level == 0:
            expr = self.parse_expr(level - 1)
            if not expr:
                return None
            # Assignment
            if self.accept('equals'):
                if not isinstance(expr, syntax.Identifier):
                    raise ParseError()
                rhs = self.parse_expr()
                return syntax.Assignment(expr, rhs)
            return expr

    def next_token(self):
        self.token = self.lexer.get_token()

def interpret(path):
    with open(path) as f:
        p = Parser(f)
        block = p.parse_block()
        p.expect('EOF')

    ctx = syntax.Context(None)
    for k, v in mg_builtins.builtins.items():
        ctx.store(k, v)

    for expr in block:
        expr.eval(ctx)

interpret(sys.argv[1])
