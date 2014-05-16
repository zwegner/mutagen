import sys

import lexer
import libparse

rule_table = [
    ['prod_group', ('LPAREN prod_expr RPAREN', lambda p: '(%s)' % p[1])],
    ['prod_opt', ('LBRACKET prod_expr RBRACKET', lambda p: '[%s]' % p[1])],
    ['prod_atom', ('(prod_group|prod_opt|IDENTIFIER) [STAR]',
        lambda p: '%s%s' % (p[0], p[1]))],
    ['prod_seq', ('prod_atom*', lambda p: ' '.join(p))],
    ['prod_expr', ('prod_seq (BIT_OR prod_seq)*',
        lambda p: '|'.join([p[0]] + [i[1] for i in p[1]]))],
    ['stmt', 'def_syntax'],
    ['stmts', 'stmt*'],
]

@libparse.rule_fn(rule_table,
    'def_syntax', 'SYNTAX IDENTIFIER COLON prod_expr SEMICOLON')
def parse_def_syntax(p):
    rule, prod = p[1], p[3]
    parser.create_rule(rule, prod, None)
    return 'syntax[%s]' % rule

parser = libparse.Parser(rule_table, 'stmts')
tokenizer = lexer.Lexer()

with open(sys.argv[1]) as f:
    tokenizer.input(f.read())
    print(parser.parse(tokenizer))
