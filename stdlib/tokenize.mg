import re

def ignore_token_fn(t):
    return Nil

def null_token_fn(t):
    return t

# A class for eating up input, matching against a lex rule
class TokenMatcher:
    def __init__(name, regex, fn):
        return make(['name', name], ['regex', regex], ['fn', fn])
    def match(self, s):
        return self.regex.match(s)

# A token from the input stream
class Token:
    def __init__(type, value):
        return make(['type', type], ['value', value])

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
        return make(['token_matchers', token_matchers])

    def tokenize_input(self, input):
        r = []
        # HACK: manually split up lines
        for i in str_split_lines(input):
            while len(i) > 0:
                best_match = [0, 0]
                best_token = Nil
                for t in self.token_matchers:
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
                if Nil != token:
                    r = r + [token]
            r = r + [Token('NEWLINE', '\n')]
        return r
