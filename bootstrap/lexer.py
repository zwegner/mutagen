import itertools
import sys

import liblex

import syntax

str_escapes = {
    'n': '\n',
    't': '\t',
    'b': '\b',
    '\\': '\\',
    '\'': '\'',
}

def process_string(t):
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
    return liblex.Token(t.type, string)

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
    'syntax',
    'union',
    'while',
    'yield',
]
token_map = {
    'AT':              r'@',
    'BIT_AND':         r'&',
    'BIT_OR':          r'\|',
    'BIT_XOR':         r'\^',
    'COLON':           r':',
    'COMMA':           r',',
    'COMMENT':         r'\#.*',
    'EQUALS':          r'=',
    'EQUALS_EQUALS':   r'==',
    'FLOORDIV':        r'//',
    'GREATER':         r'>',
    'GREATER_EQUALS':  r'>=',
    'IDENTIFIER':      (r'[a-zA-Z_][a-zA-Z0-9_]*',
        lambda t: liblex.Token(t.value.upper(), t.value) if t.value in keywords else t),
    'INTEGER':         (r'((0x[0-9a-fA-F]*)|([0-9]+))',
        lambda t: liblex.Token(t.type, int(t.value, 0))),
    'INVERSE':         r'~',
    'LBRACE':          r'{',
    'LBRACKET':        r'\[',
    'LESS':            r'<',
    'LESS_EQUALS':     r'<=',
    'LPAREN':          r'\(',
    'MINUS':           r'-',
    'MODULO':          r'%',
    'NEWLINE':         r'\n',
    'NOT_EQUALS':      r'!=',
    'PERIOD':          r'\.',
    'PLUS':            r'\+',
    'RARROW':          r'->',
    'RBRACE':          r'}',
    'RBRACKET':        r']',
    'RPAREN':          r'\)',
    'SEMICOLON':       r';',
    'SHIFT_LEFT':      r'<<',
    'SHIFT_RIGHT':     r'>>',
    'STAR':            r'\*',
    'STAR_STAR':       r'\*\*',
    'STRING':          (r'\'((\\.)|[^\\\'])*\'', process_string),
    'WHITESPACE':      r'[ \t\r]+',
}
skip = {'COMMENT'}
tokens = list(token_map.keys()) + [k.upper() for k in keywords] + ['INDENT', 'DEDENT']

def error(t, msg):
    info = t.info or liblex.Info('<unknown file>', 0)
    print('%s(%s): %s' % (info.filename, info.lineno, msg))
    sys.exit(1)

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
                yield liblex.Token('WHITESPACE', '')
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
                yield liblex.Token('INDENT', '')
            else:
                # Pop off the indent stack until we reach the previous level
                while spaces < ws_stack[-1]:
                    ws_stack.pop()
                    yield liblex.Token('DEDENT', '')
                if spaces != ws_stack[-1]:
                    error(t, 'unindent level does not match any previous indent')
        else:
            yield t

    # Make sure we have enough indents at EOF
    while len(ws_stack) > 1:
        ws_stack.pop()
        yield liblex.Token('DEDENT', '')

    yield None

class Lexer(liblex.Tokenizer):
    def __init__(self):
        super().__init__(token_map, skip)

    def input(self, input, filename=None):
        super().input(input, filename=filename)
        # Big ass chain of generators
        self.tokens = process_indentation(process_whitespace(process_newlines(
            self.tokens)))

    def token(self):
        return next(self.tokens)
