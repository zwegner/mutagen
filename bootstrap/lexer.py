import itertools
import sys

import ply.lex as lex

import syntax

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
    'BIT_AND':         r'&',
    'BIT_OR':          r'\|',
    'BIT_XOR':         r'\^',
    'COLON':           r':',
    'COMMA':           r',',
    'EQUALS':          r'=',
    'EQUALS_EQUALS':   r'==',
    'GREATER':         r'>',
    'GREATER_EQUALS':  r'>=',
    'LBRACE':          r'{',
    'LBRACKET':        r'\[',
    'LESS':            r'<',
    'LESS_EQUALS':     r'<=',
    'LPAREN':          r'\(',
    'MINUS':           r'-',
    'NOT_EQUALS':      r'!=',
    'PERIOD':          r'\.',
    'PLUS':            r'\+',
    'RBRACE':          r'}',
    'RBRACKET':        r']',
    'RPAREN':          r'\)',
    'SEMICOLON':       r';',
    'STAR':            r'\*',
}

keywords = [
    'False',
    'None',
    'True',
    'and',
    'assert',
    'break',
    'class',
    'continue',
    'def',
    'elif',
    'else',
    'for',
    'from',
    'if',
    'import',
    'in',
    'lambda',
    'not',
    'or',
    'pass',
    'return',
    'union',
    'while',
    'yield',
]

for keyword in keywords:
    tokens += [keyword.upper()]
    globals()['t_%s' % keyword.upper()] = keyword

for token, regex in token_map.items():
    tokens += [token]
    globals()['t_%s' % token] = regex

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value in keywords:
        t.type = t.value.upper()
    return t

str_escapes = {
    'n': '\n',
    't': '\t',
    'b': '\b',
    '\\': '\\',
    '\'': '\'',
}

def t_STRING(t):
    r'\'((\\.)|[^\\\'])*\''
    string = ''
    i = iter(t.value[1:-1])
    while True:
        try:
            c = next(i)
        except StopIteration:
            break
        if c == '\\':
            c = next(i)
            if c == 'x':
                string += chr(int(next(i) + next(i), 16))
            elif c not in str_escapes:
                error(t, 'bad string escape: "\%s"' % c)
            else:
                string += str_escapes[c]
        else:
            string += c
    t.value = string
    return t

def t_INTEGER(t):
    r'-?((0x[0-9a-fA-F]*)|([0-9]+))'
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
    error(t, 'invalid token: %s' % t)

def error(t, msg):
    print('%s(%s): %s' % (syntax.filename, t.lineno, msg))
    sys.exit(1)

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

def process_indentation(tokens):
    ws_stack = [0]
    for t in tokens:
        # Whitespace has been processed, so this token is at the beginning
        # of a non-empty line
        if t.type == 'WHITESPACE':
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
                    error(t, 'unindent level does not match' +
                        'any previous indent')
        else:
            yield t

    # Make sure we have enough indents at EOF
    while len(ws_stack) > 1:
        ws_stack.pop()
        yield Token('DEDENT', '')

    yield None

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
