import copy
import re

# Info means basically filename/line number, used for reporting errors
class Info:
    def __init__(self, filename, lineno):
        self.filename = filename
        self.lineno = lineno
    def __str__(self):
        return 'Info("%s", %s)' % (self.filename, self.lineno)

class Token:
    def __init__(self, type, value, info=None):
        self.type = type
        self.value = value
        self.info = info
    def copy(self, type=None, value=None, info=None):
        c = copy.copy(self)
        if type is not None:  c.type = type
        if value is not None: c.value = value
        if info is not None:  c.info = info
        return c
    def __str__(self):
        return 'Token(%s, "%s", info=%s)' % (self.type, self.value, self.info)

class Tokenizer:
    def __init__(self, token_list, skip):
        self.token_fns = {}
        # If the token list is actually a dict, sort by longest regex first
        if isinstance(token_list, dict):
            token_list = sorted(token_list.items(), key=lambda item: -len(item[1]))
        sorted_tokens = []
        for k, v in token_list:
            if isinstance(v, tuple):
                v, fn = v
                self.token_fns[k] = fn
            sorted_tokens.append([k, v])
        regex = '|'.join('(?P<%s>%s)' % (k, v) for k, v in sorted_tokens)
        self.matcher = re.compile(regex).match
        self.skip = skip

    def input(self, text, filename=None):
        match = self.matcher(text)
        lineno = 1
        tokens = []
        while match is not None:
            type = match.lastgroup
            value = match.group(type)
            if type not in self.skip:
                token = Token(type, value)
                if type in self.token_fns:
                    token = self.token_fns[type](token)
                token.info = Info(filename, lineno)
                tokens.append(token)
            lineno += value.count('\n')
            match = self.matcher(text, match.end())
        return TokenizerContext(tokens)

class TokenizerContext:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.saved_token = None

    def peek(self):
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def next(self):
        token = self.peek()
        self.pos += 1
        return token

    def accept(self, t):
        if self.peek() and self.peek().type == t:
            return self.next()
        return None

    def expect(self, t):
        if self.peek().type != t:
            raise RuntimeError('got %s instead of %s' % (self.peek().type, t))
        return self.next()
