import re

def ignore_token_fn(t):
    return None

# A token from the input stream
class Token(type, value, info=None):
    # XXX meh, shadowing
    type_ = type
    # XXX recursion
    def copy(self, type=None, value=None, info=None):
        if type == None: type = self.type
        if value == None: value = self.value
        if info == None: info = self.info
        return type_(self)(type, value, info=info)
    def __str__(self):
        return 'Token({}, {}, info={})'.format(repr(self.type), repr(self.value), self.info)

class Info(filename, lineno):
    def __str__(self):
        return 'Info({}, {})'.format(repr(self.filename), self.lineno)

class LexerContext(tokens, pos=0):
    def peek(self):
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    # XXX recursion
    def next(self):
        return [(self <- .pos = self.pos + 1), self.peek()]

    def accept(self, t):
        if self.peek() and self.peek().type == t:
            return self.next()
        return [self, None]

    def expect(self, t):
        if not self.peek() or self.peek().type != t:
            error('got {} instead of {}'.format('\n'.join(map(str, self.tokens[self.pos:self.pos+40])), t))
        return self.next()

class Lexer:
    def __init__(token_list):
        token_matchers = []
        if isinstance(token_list, dict):
            token_list = sorted(token_list, key=lambda(item): -len(item[1]))
        token_fns = {}
        for [k, v] in token_list:
            if isinstance(v, list):
                [v, fn] = v
                token_fns = token_fns + {k: fn}
            token_matchers = token_matchers + ['(?P<{}>{})'.format(k, v)]
        #token_match = re.compile('|'.join(token_matchers)).match
        token_match = re_compile_match('|'.join(token_matchers))
        return {'token_match': token_match, 'token_fns': token_fns}

    def lex_input(self, text, filename):
        lineno = 1
        start = end = 0
        m = self.token_match(text, end)
        while m:
            #type = m.get_last_group()
            #[start, end] = m.span(0)
            [type, start, end] = m
            match = text[start:end]

            token = Token(type, match, info=Info(filename, lineno))
            if type in self.token_fns:
                token = self.token_fns[type](token)
            if token != None:
                yield token
            lineno = lineno + match.count('\n')
            m = self.token_match(text, end)
        if end < len(text):
            error('Error lexing input: {}...'.format(text[end:40]))

    def input(self, text, filename=None):
        return LexerContext(list(self.lex_input(text, filename)))
