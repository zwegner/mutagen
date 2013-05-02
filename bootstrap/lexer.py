import itertools

import ply.lex as lex

class LexError(Exception):
    pass

tokens = [
    'WHITESPACE',
    'INDENT',
    'DEDENT',
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
    'pass',
    'import',
    'def',
    'lambda',
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

t_ignore = "\r"

def t_WHITESPACE(t):
    r'[ \t]+'
    return t

def t_NEWLINE(t):
    r'\n'
    t.lexer.lineno += 1
    return t

def t_COMMENT(t):
    r'\#.*'
    pass

def t_error(t):
    raise LexError('%s(%i): %s' % (syntax.filename, t.lineno, t))

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value
    def __str__(self):
        return 'Token(%s, \'%s\')' % (self.type, self.value)

def process_newlines(tokens):
    braces = brackets = parens = 0
    for t in tokens:
        if 0: pass
        elif t.type == 'LBRACE': braces += 1
        elif t.type == 'RBRACE': braces -= 1
        elif t.type == 'LBRACKET': brackets += 1
        elif t.type == 'RBRACKET': brackets -= 1
        elif t.type == 'LPAREN': parens += 1
        elif t.type == 'RPAREN': parens -= 1

        if t.type == 'NEWLINE':
            if braces == brackets == parens == 0:
                yield t
        else:
            yield t

def process_whitespace(tokens):
    after_newline = True
    # Get two copies of the token stream, and offset the second so we can get
    # (current, next) tuples for every token
    tokens, next_tokens = itertools.tee(tokens, 2)
    next(next_tokens)
    for token, next_token in itertools.zip_longest(tokens, next_tokens):
        # Check whitespace only at the beginning of lines
        if after_newline:
            # Don't generate indent/dedent on empty lines
            if token.type == 'WHITESPACE':
                if next_token.type != 'NEWLINE':
                    yield token
            # Got a token at the beginning of the line--yield empty whitespace
            elif token.type != 'NEWLINE':
                yield Token('WHITESPACE', '')
                yield token

        # Not after a newline--ignore whitespace
        elif token.type != 'WHITESPACE':
            yield token

        after_newline = token.type == 'NEWLINE'

    yield None

def process_indentation(tokens):
    ws_stack = [0]
    for t in tokens:
        # Whitespace has been processed, so this token is at the beginning
        # of a non-empty line
        if t and t.type == 'WHITESPACE':
            spaces = len(t.value.replace('\t', ' '*4))

            # Check the indent level against the stack
            if spaces > ws_stack[-1]:
                ws_stack.append(spaces)
                yield Token('INDENT', '')
            else:
                # Pop off the indent stack until we reach the previous level
                while spaces < ws_stack[-1]:
                    ws_stack.pop()
                    yield Token('DEDENT', '')
                if spaces != ws_stack[-1]:
                    raise LexError('%s(%i): unindent level does not match' +
                        'any previous indent' % (syntax.filename, t.lineno))
        else:
            yield t

class Lexer:
    def __init__(self):
        self.lexer = lex.lex()

    def input(self, input):
        self.lexer.input(input)
        # Big ass chain of generators
        self.stream = process_indentation(process_whitespace(process_newlines(
            self.gen_tokens())))

    def gen_tokens(self):
        while True:
            t = self.lexer.token()
            if t is None:
                break
            yield t

    def token(self):
        return next(self.stream)

def get_lexer():
    return Lexer()
