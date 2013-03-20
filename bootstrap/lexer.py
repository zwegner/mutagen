import ply.lex as lex

class LexError(Exception):
    pass

tokens = (
    'EQUALS',
    'LPAREN',
    'RPAREN',
    'LBRACKET',
    'RBRACKET',
    'LBRACE',
    'RBRACE',
    'COMMA',
    'PERIOD',
    'SEMICOLON',
    'NEWLINE',
    'IDENTIFIER',
    'IMPORT',
    'STRING',
)

t_EQUALS       = r'='
t_LPAREN       = r'\('
t_RPAREN       = r'\)'
t_LBRACKET     = r'\['
t_RBRACKET     = r']'
t_LBRACE       = r'{'
t_RBRACE       = r'}'
t_COMMA        = r','
t_PERIOD       = r'\.'
t_SEMICOLON    = r';'

keywords = (
'import',
)

str_escapes = {
    'n': '\n',
    't': '\t',
    'b': '\b',
}

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value in keywords:
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

# Ignored characters
t_ignore = " \t"

def t_NEWLINE(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")
    return t

def t_COMMENT(t):
    r'\#.*'
    pass

def t_error(t):
    raise LexError()

def get_lexer(input):
    l = lex.lex()
    lex.input(input.read())
    return lex
