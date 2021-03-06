import liblex

str_escapes = {
    'n': '\n',
    't': '\t',
    'b': '\b',
    '\\': '\\',
    '\'': '\'',
}

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

# Special token-transforming functions
def t_identifier(t):
    for k in keywords:
        if t.value == k:
            return t <- .type = k.upper()
    return t

def t_integer(t):
    return t <- .value = int(t.value, 0)

def t_string(t):
    i = 1
    result = ''
    while i < len(t.value) - 1:
        c = t.value[i]
        if c == '\\':
            i = i + 1
            c = t.value[i]
            esc = None
            if c in str_escapes:
                esc = str_escapes[c]
            else:
                error('bad string escape: "\\'+c+'"')
            result = result + esc
        else:
            result = result + c
        i = i + 1
    return liblex.Token(t.type, result)

token_map = {
    'AT':              '@',
    'BIT_AND':         '&',
    'BIT_OR':          '\\|',
    'BIT_XOR':         '\\^',
    'COLON':           ':',
    'COMMA':           ',',
    'EQUALS':          '=',
    'EQUALS_EQUALS':   '==',
    'FLOORDIV':        '//',
    'GREATER':         '>',
    'GREATER_EQUALS':  '>=',
    'INVERSE':         '~',
    'LARROW':          '<-',
    'LBRACE':          '{',
    'LBRACKET':        '\\[',
    'LESS':            '<',
    'LESS_EQUALS':     '<=',
    'LPAREN':          '\\(',
    'MINUS':           '-',
    'MODULO':          '%',
    'NEWLINE':         '\n',
    'NOT_EQUALS':      '!=',
    'PERIOD':          '\\.',
    'PLUS':            '\\+',
    'RBRACE':          '}',
    'RBRACKET':        ']',
    'RPAREN':          '\\)',
    'SEMICOLON':       ';',
    'SHIFT_LEFT':      '<<',
    'SHIFT_RIGHT':     '>>',
    'STAR':            '\\*',
    'STAR_STAR':       '\\*\\*',
    'WHITESPACE':      '[ \t]+',
    'IDENTIFIER':      ['[a-zA-Z_][a-zA-Z0-9_]*', t_identifier],
    'INTEGER':         ['((0x[0-9a-fA-F]*)|([0-9]+))', t_integer],
    'COMMENT':         ['#[^\n]*', liblex.ignore_token_fn],
    'STRING':          ['\'((\\\\.)|[^\\\\\'])*\'', t_string],
}

def process_newlines(tokens):
    braces = brackets = parens = 0
    for t in tokens:
        if t.type == 'LBRACE':     braces = braces + 1
        elif t.type == 'RBRACE':   braces = braces - 1
        elif t.type == 'LBRACKET': brackets = brackets + 1
        elif t.type == 'RBRACKET': brackets = brackets - 1
        elif t.type == 'LPAREN':   parens = parens + 1
        elif t.type == 'RPAREN':   parens = parens - 1

        if t.type == 'NEWLINE':
            if braces == 0 and brackets == 0 and parens == 0:
                yield t
        else:
            yield t

def process_whitespace(tokens):
    after_newline = True
    # HACK until we have some sliding-window-type generator
    tokens = list(tokens)
    for [token, next_token] in zip(tokens, tokens[1:] + [None]):
        # Check whitespace only at the beginning of lines
        if after_newline:
            # Don't generate indent/dedent on empty lines
            if token.type == 'WHITESPACE':
                if next_token != None and next_token.type != 'NEWLINE':
                    yield token
            # Got a token at the beginning of the line--yield empty whitespace
            elif token.type != 'NEWLINE':
                yield liblex.Token('WHITESPACE', '')
                yield token

        # Not after a newline--ignore whitespace
        elif token.type != 'WHITESPACE':
            yield token

        after_newline = (token.type == 'NEWLINE')

def process_indentation(tokens):
    ws_stack = [0]
    for t in tokens:
        # Whitespace has been processed, so this token is at the beginning
        # of a non-empty line
        if t.type == 'WHITESPACE':
            spaces = len(t.value) #.replace('\t', ' '*4))

            # Check the indent level against the stack
            if spaces > ws_stack[-1]:
                ws_stack = ws_stack + [spaces]
                yield liblex.Token('INDENT', '')
            else:
                # Pop off the indent stack until we reach the previous level
                while spaces < ws_stack[-1]:
                    ws_stack = ws_stack[:len(ws_stack) - 1]
                    yield liblex.Token('DEDENT', '')
                if spaces != ws_stack[-1]:
                    error('unindent level does not match' +
                        'any previous indent')
        else:
            yield t

    # Make sure we have enough indents at EOF
    while len(ws_stack) > 1:
        ws_stack = ws_stack[:len(ws_stack) - 1]
        yield liblex.Token('DEDENT', '')

lexer = liblex.Lexer(token_map)

def lex_input(text, filename=None):
    tokens = lexer.lex_input(text, filename)
    tokens = process_newlines(tokens)
    tokens = process_whitespace(tokens)
    tokens = process_indentation(tokens)
    return tokens

def input(text, filename=None):
    return liblex.LexerContext(list(lex_input(text, filename=filename)))
