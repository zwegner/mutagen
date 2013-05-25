import tokenize

str_escapes = {
    'n': '\n',
    't': '\t',
    'b': '\b',
    '\\': '\\',
    '\'': '\'',
}

# Special token-transforming functions
def t_identifier(t):
    r = t
    for k in keywords:
        if t.value == k:
            r = tokenize.Token(str_upper(k), t.value)
    return r

def t_integer(t):
    return tokenize.Token(t.type, parse_int(t.value, 0))

def t_string(t):
    i = 1
    result = ''
    while i < len(t.value) - 1:
        c = t.value[i]
        if c == '\\':
            i = i + 1
            c = t.value[i]
            esc = Nil
            if c in str_escapes:
                esc = str_escapes[c]
            else:
                print('bad string escape: "\\'+c+'"')
                error
            result = result + esc
        else:
            result = result + c
        i = i + 1
    return tokenize.Token(t.type, result)

token_map = [
    ['COLON',           ':'],
    ['COMMA',           ','],
    ['EQUALS',          '='],
    ['EQUALS_EQUALS',   '=='],
    ['GREATER',         '>'],
    ['GREATER_EQUALS',  '>='],
    ['LBRACE',          '{'],
    ['LBRACKET',        '\\['],
    ['LESS',            '<'],
    ['LESS_EQUALS',     '<='],
    ['LPAREN',          '\\('],
    ['MINUS',           '-'],
    ['NOT_EQUALS',      '!='],
    ['PERIOD',          '\\.'],
    ['PLUS',            '\\+'],
    ['RBRACE',          '}'],
    ['RBRACKET',        ']'],
    ['RPAREN',          '\\)'],
    ['SEMICOLON',       ';'],
    ['STAR',            '\\*'],
    ['WHITESPACE',      '[ \t]+'],
    ['IDENTIFIER',      '[a-zA-Z_][a-zA-Z0-9_]*', t_identifier],
    ['INTEGER',         '-?((0x[0-9a-fA-F]*)|([0-9]+))', t_integer],
    ['COMMENT',         '#.*', tokenize.ignore_token_fn],
    ['STRING',          '\'((\\\\.)|[^\\\\\'])*\'', t_string],
]

keywords = [
    'Nil',
    'and',
    'class',
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
    'while',
    'yield',
]

def process_newlines(tokens):
    new_tokens = []
    braces = 0
    brackets = 0
    parens = 0
    for t in tokens:
        if t.type == 'LBRACE':     braces = braces + 1
        elif t.type == 'RBRACE':   braces = braces - 1
        elif t.type == 'LBRACKET': brackets = brackets + 1
        elif t.type == 'RBRACKET': brackets = brackets - 1
        elif t.type == 'LPAREN':   parens = parens + 1
        elif t.type == 'RPAREN':   parens = parens - 1

        if t.type == 'NEWLINE':
            if braces == 0 and brackets == 0 and parens == 0:
                new_tokens = new_tokens + [t]
        else:
            new_tokens = new_tokens + [t]
    return new_tokens

def process_whitespace(tokens):
    after_newline = 1
    i = 0
    new_tokens = []
    while i < len(tokens):
        token = tokens[i]
        if i < len(tokens) - 1:
            next_token = tokens[i+1]
        else:
            next_token = Nil

        # Check whitespace only at the beginning of lines
        if after_newline:
            # Don't generate indent/dedent on empty lines
            if token.type == 'WHITESPACE':
                if next_token != Nil and next_token.type != 'NEWLINE':
                    new_tokens = new_tokens + [token]
            # Got a token at the beginning of the line--yield empty whitespace
            elif token.type != 'NEWLINE':
                new_tokens = new_tokens + [tokenize.Token('WHITESPACE', ''), token]

        # Not after a newline--ignore whitespace
        elif token.type != 'WHITESPACE':
            new_tokens = new_tokens + [token]

        after_newline = (token.type == 'NEWLINE')

        i = i + 1

    return new_tokens

def process_indentation(tokens):
    ws_stack = [0]
    new_tokens = []
    for t in tokens:
        # Whitespace has been processed, so this token is at the beginning
        # of a non-empty line
        if t.type == 'WHITESPACE':
            spaces = len(t.value) #.replace('\t', ' '*4))

            # Check the indent level against the stack
            if spaces > ws_stack[-1]:
                ws_stack = ws_stack + [spaces]
                new_tokens = new_tokens + [tokenize.Token('INDENT', '')]
            else:
                # Pop off the indent stack until we reach the previous level
                while spaces < ws_stack[-1]:
                    ws_stack = slice(ws_stack, 0, len(ws_stack) - 1)
                    new_tokens = new_tokens + [tokenize.Token('DEDENT', '')]
                if spaces != ws_stack[-1]:
                    error(t, 'unindent level does not match' +
                        'any previous indent')
        else:
            new_tokens = new_tokens + [t]

    # Make sure we have enough indents at EOF
    while len(ws_stack) > 1:
        ws_stack = slice(ws_stack, 0, len(ws_stack) - 1)
        new_tokens = new_tokens + [tokenize.Token('DEDENT', '')]

    return new_tokens

tokenizer = tokenize.Tokenizer(token_map)

def lex_input(input):
    tokens = tokenizer.tokenize_input(input)
    tokens = process_newlines(tokens)
    tokens = process_whitespace(tokens)
    tokens = process_indentation(tokens)
    return tokens
