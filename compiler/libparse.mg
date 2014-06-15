import liblex

# ParseResult works like a tuple for the results of parsed rules, but with an
# additional .get_info(n) method for getting line-number information out
class ParseResult(items, info):
    def __getitem__(self, n):
        return self.items[n]
    def get_info(self, n):
        return self.info[n]

class Context(fn_table, tokenizer):
    pass

# Classes to represent grammar structure. These are hierarchically nested, and
# operate through the parse method, usually calling other rules' parse methods.

# Parse either a token or a nonterminal of the grammar
class Identifier(name: str):
    def parse(self, fn_table, tokenizer):
        if self.name in fn_table:
            return fn_table[self.name].parse(fn_table, tokenizer)
        # XXX check token name validity
        [tokenizer, token] = tokenizer.accept(self.name)
        if token:
            return [tokenizer, token.value, token.info]
        return None
    def __str__(self):
        return '"{}"'.format(self.name)

# Parse a rule repeated at least <min> number of times (used for * and + in EBNF)
class Repeat(item, min=0):
    def parse(self, fn_table, tokenizer):
        results = []
        result = self.item.parse(fn_table, tokenizer)
        while result:
            [tokenizer, token, _] = result
            results = results + [token]
            result = self.item.parse(fn_table, tokenizer)
        if len(results) >= self.min:
            return [tokenizer, results, None]
        return None
    def __str__(self):
        return 'rep({})'.format(self.item)

# Parse a sequence of multiple consecutive rules
class Sequence(items):
    def parse(self, fn_table, tokenizer):
        results = []
        infos = []
        for item in self.items:
            result = item.parse(fn_table, tokenizer)
            if not result:
                return None
            [tokenizer, token, info] = result
            results = results + [token]
            infos = infos + [info]
        return [tokenizer, results, infos]
    def __str__(self):
        return 'seq({})'.format(','.join(map(str, self.items)))

# Parse one of a choice of multiple rules
class Alternation(items):
    def parse(self, fn_table, tokenizer):
        for item in self.items:
            result = item.parse(fn_table, tokenizer)
            if result:
                return result
        return None
    def __str__(self):
        return 'alt({})'.format(','.join(map(str, self.items)))

# Either parse a rule or not
class Optional(item):
    def parse(self, fn_table, tokenizer):
        result = self.item.parse(fn_table, tokenizer)
        return result or [None, None]
    def __str__(self):
        return 'opt({})'.format(self.item)

# Parse a and then call a user-defined function on the result
class FnWrapper:
    def __init__(prod, fn):
        # Make sure top-level rules are a sequence. When we pass parse results
        # to the user-defined function, it must be returned in an array, so we
        # can use the ParserResults class and have access to the parse info
        if not isinstance(prod, Sequence):
            prod = Sequence([prod])
        return {'prod': prod, 'fn': fn}
    def parse(self, fn_table, tokenizer):
        result = self.prod.parse(fn_table, tokenizer)
        if result:
            [tokenizer, result, info] = result
            return [tokenizer, self.fn(ParseResult(result, info)), None]
        return None
    def __str__(self):
        return str(self.prod)

# Mini parser for our grammar specification language (basically EBNF)

# After either a parenthesized group or an identifier, we accept * and + for
# repeating the aforementioned item (either zero or more times, or one or more)
def parse_repeat(tokenizer, repeated):
    [tokenizer, token] = tokenizer.accept('STAR')
    if token:
        return [tokenizer, Repeat(repeated)]
    else:
        [tokenizer, token] = tokenizer.accept('PLUS')
        if token:
            return [tokenizer, Repeat(repeated, min=1)]
    return [tokenizer, repeated]

def parse_rule_atom(parse_rule_expr, tokenizer):
    # Parenthesized rules: just used for grouping
    [tokenizer, token] = tokenizer.accept('LPAREN')
    if token:
        [tokenizer, result] = parse_rule_expr(parse_rule_expr, tokenizer)
        [tokenizer, _] = tokenizer.expect('RPAREN')
        [tokenizer, result] = parse_repeat(tokenizer, result)
    # Bracketed rules are entirely optional
    else:
        [tokenizer, token] = tokenizer.accept('LBRACKET')
        if token:
            result = Optional(parse_rule_expr(parse_rule_expr, tokenizer))
            [tokenizer, _] = tokenizer.expect('RBRACKET')
        # Otherwise, it must be a regular identifier
        else:
            [tokenizer, token] = tokenizer.accept('IDENTIFIER')
            [tokenizer, result] = parse_repeat(tokenizer, Identifier(token.value))
    return [tokenizer, result]

# Parse the concatenation of one or more expressions
def parse_rule_seq(parse_rule_expr, tokenizer):
    items = []
    token = tokenizer.peek()
    while (token and token.type != 'RBRACKET' and
            token.type != 'RPAREN' and token.type != 'PIPE'):
        [tokenizer, result] = parse_rule_atom(parse_rule_expr, tokenizer)
        items = items + [result]
        token = tokenizer.peek()
    # Only return a sequence if there's multiple items, otherwise there's way
    # too many [0]s when extracting parsed items in complicated rules
    if len(items) > 1:
        return [tokenizer, Sequence(items)]
    return [tokenizer, items[0]]

# Top-level parser, parse any number of sequences, joined by the alternation
# operator, |
def parse_rule_expr(parse_rule_expr, tokenizer):
    [tokenizer, seq] = parse_rule_seq(parse_rule_expr, tokenizer)
    items = [seq]
    [tokenizer, token] = tokenizer.accept('PIPE')
    while token:
        [tokenizer, seq] = parse_rule_seq(parse_rule_expr, tokenizer)
        items = items + [seq]
        [tokenizer, token] = tokenizer.accept('PIPE')
    if len(items) > 1:
        return [tokenizer, Alternation(items)]
    return [tokenizer, items[0]]

# ...And a mini lexer too

rule_tokens = {
    'IDENTIFIER': '[a-zA-Z_]+',
    'LBRACKET':   '\\[',
    'LPAREN':     '\\(',
    'PIPE':       '\\|',
    'RBRACKET':   '\\]',
    'RPAREN':     '\\)',
    'STAR':       '\\*',
    'PLUS':       '\\+',
    'WHITESPACE': [' ', lambda(p) { return None; }],
}
rule_lexer = liblex.Lexer(rule_tokens)

class Parser:
    def create_rule(fn_table, rule, prod, fn):
        [_, prod] = parse_rule_expr(parse_rule_expr, rule_lexer.input(prod))
        if fn != None:
            prod = FnWrapper(prod, fn)
        if rule not in fn_table:
            fn_table = fn_table + {rule: Alternation([prod])}
        else:
            prev_items = fn_table[rule].items
            fn_table = fn_table + {rule: Alternation(prev_items + [prod])}
        return fn_table

    def __init__(rule_table, start):
        fn_table = {}
        for rp in rule_table:
            [rule, prods] = [rp[0], rp[1:]]
            for prod in prods:
                fn = None
                if isinstance(prod, list):
                    [prod, fn] = prod
                fn_table = create_rule(fn_table, rule, prod, fn)
        return {'fn_table': fn_table, 'start': start}

    def parse(self, tokenizer):
        prod = self.fn_table[self.start]
        result = prod.parse(self.fn_table, tokenizer)
        if not result:
            error('bad parse near token {}'.format(tokenizer.peek()))
        [tokenizer, result, info] = result
        if tokenizer.peek() != None:
            error('parser did not consume entire input, near token {}'.format(
                tokenizer.peek()))
        return result
