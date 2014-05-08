import re

def ignore_token_fn(t):
    return None

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
        token_fns = {}
        for token in token_map:
            if len(token) > 2:
                [type, regex, fn] = token
                token_fns = token_fns + {type: fn}
            else:
                [type, regex] = token
            token_matchers = token_matchers + ['(?P<{}>{})'.format(type, regex)]
        token_match = re.compile('|'.join(token_matchers)).match
        return {'token_match': token_match, 'token_fns': token_fns}

    def tokenize_input(self, input):
        while len(input) > 0:
            m = self.token_match(input)
            if m != None:
                type = m.get_last_group()
            else:
                error('Error lexing input: {}...'.format(slice(input, 40)))

            [start, end] = m.span(0)
            match = slice(input, end)
            input = slice(input, end, None)

            token = Token(type, match)
            if type in self.token_fns:
                token = self.token_fns[type](token)
            if token != None:
                yield token
