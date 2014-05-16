import re

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value
    def __str__(self):
        return 'Token(%s, "%s")' % (self.type, self.value)

class Tokenizer:
    def __init__(self, token_map, skip):
        new_token_map = {}
        self.token_fns = {}
        for k, v in token_map.items():
            if isinstance(v, tuple):
                v, fn = v
                self.token_fns[k] = fn
            new_token_map[k] = v
        # Sort by longest regex first
        sorted_keys = sorted(new_token_map.items(), key=lambda item: -len(item[1]))
        regex = '|'.join('(?P<%s>%s)' % (k, v) for k, v in sorted_keys)
        self.matcher = re.compile(regex).match
        self.skip = skip

    def input(self, line):
        self.tokens = []
        match = self.matcher(line)
        while match is not None:
            type = match.lastgroup
            value = match.group(type)
            if type not in self.skip:
                token = Token(type, value)
                if type in self.token_fns:
                    token = self.token_fns[type](token)
                self.tokens.append(token)
            match = self.matcher(line, match.end())
        self.tokens.append(Token('EOF', 'EOF'))

    def peek(self):
        return self.tokens[0]

    def next(self):
        return self.tokens.pop(0)

    def accept(self, t):
        if self.peek().type == t:
            return self.next()
        return None

    def expect(self, t):
        if self.peek().type != t:
            raise RuntimeError('got %s instead of %s' % (self.peek().type, t))
        return self.next()
