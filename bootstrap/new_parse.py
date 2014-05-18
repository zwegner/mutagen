import sys

import lexer
import libparse

rule_table = [
    ['stmt', 'def_syntax'],
    ['stmts', 'stmt*'],
]

@libparse.rule_fn(rule_table,
    'def_syntax', 'SYNTAX IDENTIFIER COLON STRING SEMICOLON')
def parse_def_syntax(p):
    rule, prod = p[1], p[3]
    parser.create_rule(rule, prod, None)
    return 'syntax[%s]' % rule

parser = libparse.Parser(rule_table, 'stmts')
tokenizer = lexer.Lexer()

with open(sys.argv[1]) as f:
    tokenizer.input(f.read())
    print(parser.parse(tokenizer))
