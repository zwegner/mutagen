import re

def ignore_token_fn(t):
    return None

def null_token_fn(t):
    return t

# A class for eating up input, matching against a lex rule
class TokenMatcher(name, regex, fn):
    def match(self, s):
        return self.regex.match(s)

# A token from the input stream
class Token(type, value):
    pass

class Tokenizer:
    def __init__(token_map):
        token_matchers = []
        for token in token_map:
            if len(token) > 2:
                [type, regex, token_fn] = token
            else:
                [type, regex] = token
                token_fn = null_token_fn
            token_matchers = token_matchers + [TokenMatcher(type,
                re.parse(regex), token_fn)]
        return {'token_matchers': token_matchers}

    def tokenize_input(self, input):
        # HACK: manually split up lines
        for i in str_split_lines(input):
            while len(i) > 0:
                best_match = [False, 0]
                best_token = None
                for t in self.token_matchers:
                    m = t.match(i)
                    if m[0] and m[1] > best_match[1]:
                        best_match = m
                        best_token = t
                if not best_match[0]:
                    error('Error: '+repr(i))

                match = slice(i, 0, best_match[1])
                i = slice(i, best_match[1], len(i))
                token = best_token.fn(Token(best_token.name, match))
                if token != None:
                    yield token
            yield Token('NEWLINE', '\n')
