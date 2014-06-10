import re

def ignore_token_fn(t):
    return None

# A token from the input stream
class Token(type, value):
    pass

class Lexer:
    def __init__(token_list):
        token_matchers = []
        if isinstance(token_list, dict):
            token_list = sorted(token_list, key=lambda(item){ return -len(item[1]); })
        token_fns = {}
        for [k, v] in token_list:
            if isinstance(v, list):
                [v, fn] = v
                token_fns = token_fns + {k: fn}
            token_matchers = token_matchers + ['(?P<{}>{})'.format(k, v)]
        token_match = re.compile('|'.join(token_matchers)).match
        return {'token_match': token_match, 'token_fns': token_fns}

    def tokenize_input(self, input):
        while input:
            m = self.token_match(input)
            if m != None:
                type = m.get_last_group()
            else:
                error('Error lexing input: {}...'.format(input[:40]))

            [start, end] = m.span(0)
            [match, input] = [input[:end], input[end:]]

            token = Token(type, match)
            if type in self.token_fns:
                token = self.token_fns[type](token)
            if token != None:
                yield token
