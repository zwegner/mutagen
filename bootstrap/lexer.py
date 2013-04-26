import ply.lex as lex

class LexError(Exception):
    pass

tokens = [
    'NEWLINE',
    'IDENTIFIER',
    'STRING',
]

token_map = {
    'EQUALS':       r'=',
    'LPAREN':       r'\(',
    'RPAREN':       r'\)',
    'LBRACKET':     r'\[',
    'RBRACKET':     r']',
    'LBRACE':       r'{',
    'RBRACE':       r'}',
    'COMMA':        r',',
    'PERIOD':       r'\.',
    'SEMICOLON':    r';',
}

keywords = [
    'import',
]

for keyword in keywords:
    tokens += [keyword.upper()]
    globals()['t_%s' % keyword.upper()] = keyword

for token, regex in token_map.items():
    tokens += [token]
    globals()['t_%s' % token] = regex

str_escapes = {
    'n': '\n',
    't': '\t',
    'b': '\b',
}

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value.upper() in tokens:
        t.type = t.value.upper()
    return t

def t_STRING(t):
    r'\'[^\']*\''
    string = ''
    i = iter(t.value[1:-1])
    while True:
        try:
            c = next(i)
        except StopIteration:
            break
        if c == '\\':
            c = next(i)
            if c not in str_escapes:
                raise LexError()
            string += str_escapes[c]
        else:
            string += c
    t.value = string
    return t

t_ignore = " \t\r"

def t_NEWLINE(t):
    r'\n'
    t.lexer.lineno += 1
    return t

def t_COMMENT(t):
    r'\#.*'
    pass

def t_error(t):
    print('%s(%i): %s' % ('', t.lineno, t))
    raise LexError()

def get_lexer():
    l = lex.lex()
    return lex
