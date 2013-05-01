import ply.lex as lex

class LexError(Exception):
    pass

tokens = [
    'NEWLINE',
    'IDENTIFIER',
    'STRING',
    'INTEGER',
]

token_map = {
    'EQUALS':          r'=',
    'EQUALS_EQUALS':   r'==',
    'LPAREN':          r'\(',
    'RPAREN':          r'\)',
    'LBRACKET':        r'\[',
    'RBRACKET':        r']',
    'LBRACE':          r'{',
    'RBRACE':          r'}',
    'COMMA':           r',',
    'PLUS':            r'\+',
    'PERIOD':          r'\.',
    'SEMICOLON':       r';',
}

keywords = [
    'import',
    'def',
    'if',
    'elif',
    'else',
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

def t_INTEGER(t):
    r'(0x[0-9a-fA-F]*)|([0-9]+)'
    t.value = int(t.value, 0)
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
    print('%s(%i): %s' % (syntax.filename, t.lineno, t))
    raise LexError()

def gen_tokens(l):
    while True:
        t = l.token()
        if t is None:
            break
        yield t

def process_newlines(tokens):
    braces = brackets = parens = 0
    for t in tokens:
        if 0: pass
        #elif t.type == 'LBRACE': braces += 1
        #elif t.type == 'RBRACE': braces -= 1
        elif t.type == 'LBRACKET': brackets += 1
        elif t.type == 'RBRACKET': brackets -= 1
        elif t.type == 'LPAREN': parens += 1
        elif t.type == 'RPAREN': parens -= 1

        if t.type == 'NEWLINE':
            if braces == brackets == parens == 0:
                yield t
        else:
            yield t

class Lexer:
    def __init__(self):
        self.lexer = lex.lex()

    def input(self, input):
        self.lexer.input(input)
        self.stream = process_newlines(gen_tokens(self.lexer))

    def token(self):
        try:
            return next(self.stream)
        except StopIteration:
            return None

def get_lexer():
    return Lexer()
