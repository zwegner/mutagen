import lexer from '../lexer.mg'

for path in ['../stdlib/__builtins__.mg', '../stdlib/re.mg', '../lexer.mg']:
    blah = read_file(path)
    for t in lexer.lex_input(blah):
        print(t.type + ': ' + repr(t.value))
