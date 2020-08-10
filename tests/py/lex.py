#!/usr/bin/env python3
import utils

from bootstrap import lexer

for path in ['../../stdlib/__builtins__.mg', '../../stdlib/re.mg', '../../compiler/lexer.mg']:
    with open(path) as f:
        blah = f.read()
    lex = lexer.Lexer()
    lex = lex.input(blah)
    while True:
        t = lex.next()
        if not t:
            break
        print(t.type + ': ' + repr(t.value))
