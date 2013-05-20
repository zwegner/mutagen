#!/usr/bin/env python3
import utils

lexer = utils.import_path('../bootstrap/lexer.py')

for path in ['../stdlib/__builtins__.mg', '../stdlib/re.mg', '../lexer.mg']:
    with open(path) as f:
        blah = f.read()
    lex = lexer.get_lexer()
    lex.input(blah)
    while True:
        t = lex.token()
        if not t:
            break
        print(t.type + ': ' + repr(t.value))
