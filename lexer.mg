import re

str_escapes = [
    ['n', '\n'],
    ['t', '\t'],
    ['b', '\b'],
    ['\\', '\\'],
    ['\'', '\''],
]

# Special token-transforming functions
def t_identifier(t):
    r = t
    for k in keywords:
        if t.value == k:
            r = Token(k, t.value)
    r

def t_integer(t):
    Token(t.type, parse_int(t.value, 0))

def t_string(t):
    i = 1
    result = ''
    while i < len(t.value) - 1:
        c = t.value[i]
        if c == '\\':
            i = i + 1
            c = t.value[i]
            esc = Nil
            for kv in str_escapes:
                if c == kv[0]:
                    esc = kv[1]
            if Nil == esc:
                print('bad string escape: "\\'+c+'"')
                error
            result = result + esc
        else:
            result = result + c
        i = i + 1
    Token(t.type, result)

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
    ['WHITESPACE',      '#.*'],
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
    'while',
]

# A class for eating up input, matching against a lex rule
class TokenMatcher:
    def __init__(name, regex, fn):
        make(['name', name], ['regex', regex], ['fn', fn])
    def match(self, s):
        self.regex.match(s)

# A token from the input stream
class Token:
    def __init__(type, value):
        make(['type', type], ['value', value])

def null_token_fn(t):
    t

# Create token matchers
token_matchers = []
for token in token_map:
    type = token[0]
    regex = token[1]
    if len(token) > 2:
        token_fn = token[2]
    else:
        token_fn = null_token_fn
    token_matchers = token_matchers + [TokenMatcher(type,
        re.parse(regex), token_fn)]

def tokenize_input(input):
    r = []
    # HACK: manually split up lines
    for i in str_split_lines(input):
        while len(i) > 0:
            best_match = [0, 0]
            best_token = Nil
            for t in token_matchers:
                m = t.match(i)
                if m[0] and m[1] > best_match[1]:
                    best_match = m
                    best_token = t
            if not best_match[0]:
                print('Error: '+repr(i))
                error

            match = slice(i, 0, best_match[1])
            i = slice(i, best_match[1], len(i))
            token = best_token.fn(Token(best_token.name, match))
            r = r + [token]
        r = r + [Token('NEWLINE', '\n')]
    r
