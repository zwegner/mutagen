import itertools
import sys

import sprdpl.lex as liblex

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
    return t.copy(value=string)

keywords = [
    'False',
    'None',
    'True',
    'and',
    'as',
    'assert',
    'break',
    'class',
    'consume',
    'continue',
    'def',
    'effect',
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
    'perform',
    'resume',
    'return',
    'union',
    'while',
    'yield',
]
token_map = {
    'AT':              r'@',
    'BIT_AND':         r'&',
    'BIT_AND_EQUALS':  r'&=',
    'BIT_OR':          r'\|',
    'BIT_OR_EQUALS':   r'\|=',
    'BIT_XOR':         r'\^',
    'BIT_XOR_EQUALS':  r'\^=',
    'COLON':           r':',
    'COMMA':           r',',
    'COMMENT':         (r'\#.*', lambda t: None),
    'EQUALS':          r'=',
    'EQUALS_EQUALS':   r'==',
    'FLOORDIV':        r'//',
    'FLOORDIV_EQUALS': r'//=',
    'GREATER':         r'>',
    'GREATER_EQUALS':  r'>=',
    'IDENTIFIER':      (r'[a-zA-Z_][a-zA-Z0-9_]*',
        lambda t: t.copy(type=t.value.upper()) if t.value in keywords else t),
    'INTEGER':         (r'((0x[0-9a-fA-F]*)|([0-9]+))',
        lambda t: t.copy(value=int(t.value, 0))),
    'INVERSE':         r'~',
    'LARROW':          r'<-',
    'LBRACE':          r'{',
    'LBRACKET':        r'\[',
    'LESS':            r'<',
    'LESS_EQUALS':     r'<=',
    'LPAREN':          r'\(',
    'MINUS':           r'-',
    'MINUS_EQUALS':    r'-=',
    'MODULO':          r'%',
    'MODULO_EQUALS':   r'%=',
    'NEWLINE':         r'\n',
    'NOT_EQUALS':      r'!=',
    'PERIOD':          r'\.',
    'PLUS':            r'\+',
    'PLUS_EQUALS':     r'\+=',
    'RARROW':          r'->',
    'RBRACE':          r'}',
    'RBRACKET':        r']',
    'RPAREN':          r'\)',
    'SEMICOLON':       r';',
    'SHIFT_LEFT':      r'<<',
    'SHIFT_LEFT_EQUALS': r'<<=',
    'SHIFT_RIGHT':     r'>>',
    'SHIFT_RIGHT_EQUALS': r'>>=',
    'STAR':            r'\*',
    'STAR_EQUALS':     r'\*=',
    'STAR_STAR':       r'\*\*',
    'STAR_STAR_EQUALS':r'\*\*=',
    'STRING':          (r'\'((\\.)|[^\\\'])*\'', process_string),
    'WHITESPACE':      r'[ \t\r]+',
}

def error(t, msg):
    raise liblex.LexError(msg, info=t.info)

# First token preprocessing step: remove all newlines within braces/brackets/parentheses
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

# Second token preprocessing step: remove all whitespace tokens except the ones at the beginning
# of non-empty lines (those are the only ones that are semantically meaningful), and remove
# double newlines (unless in interactive mode and at the end of the token stream)
def process_whitespace(tokens, interactive=False):
    after_newline = True
    after_second_newline = False
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
                yield token.copy(type='WHITESPACE', value='')
                yield token

        # Not after a newline--ignore whitespace
        elif token.type != 'WHITESPACE':
            yield token

        after_second_newline = after_newline
        after_newline = token.type == 'NEWLINE'

    # HACK for interactive mode. We normally eat consecutive newlines, but in interactive mode we
    # need to check if the end of the stream has two newlines to end an indented block
    if interactive and after_newline and after_second_newline:
        yield token

# Third token preprocessing step: count the amount of whitespace at the beginning of each line,
# and generate INDENT/DEDENT tokens when it increases/decreases
def process_indentation(tokens, interactive=False):
    ws_stack = [0]
    second_last_token_type = last_token_type = None
    for token in tokens:
        second_last_token_type = last_token_type
        last_token_type = token.type
        # Whitespace has been processed, so this token is at the beginning
        # of a non-empty line
        if token.type == 'WHITESPACE':
            spaces = len(token.value.replace('\t', ' '*4))

            # Check the indent level against the stack
            if spaces > ws_stack[-1]:
                ws_stack.append(spaces)
                yield token.copy(type='INDENT', value='')
            else:
                # Pop off the indent stack until we reach the previous level
                while spaces < ws_stack[-1]:
                    ws_stack.pop()
                    yield token.copy(type='DEDENT', value='')
                if spaces != ws_stack[-1]:
                    error(token, 'unindent level does not match '
                        'any previous indent')
        else:
            yield token

    # If we're in interactive mode, don't generate the trailing DEDENT tokens unless the
    # user has entered two consecutive newlines
    if interactive and (last_token_type != 'NEWLINE' or second_last_token_type != 'NEWLINE'):
        return

    # Make sure we have enough indents at EOF
    while len(ws_stack) > 1:
        ws_stack.pop()
        yield token.copy(type='DEDENT', value='')

class Lexer(liblex.Lexer):
    def __init__(self):
        super().__init__(token_map)

    def input(self, text, filename=None, interactive=False):
        tokens = self.lex_input(text, filename)
        # Big ass chain of generators
        tokens = process_newlines(tokens)
        tokens = process_whitespace(tokens, interactive=interactive)
        tokens = process_indentation(tokens, interactive=interactive)
        return liblex.LexerContext(text, tokens, filename)
