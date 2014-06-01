import liblex

def unzip(lists):
    if not lists:
        return [], []
    return list(list(t) for t in zip(*lists))

class Context:
    def __init__(self, fn_table, tokenizer):
        self.fn_table = fn_table
        self.tokenizer = tokenizer
        self.check_prods = set()

class ParseResult:
    def __init__(self, items, info):
        self.items = items
        self.info = info
    def __getitem__(self, n):
        return self.items[n]
    def get_info(self, n):
        if self.info:
            return self.info[n]
        return None

# Dummy sentinel object
BAD_PARSE = object()

# Classes to represent grammar structure

class String:
    def __init__(self, rule, name):
        self.rule = rule
        self.name = name
    def parse(self, ctx):
        if self.name in ctx.fn_table:
            return ctx.fn_table[self.name].parse(ctx)
        elif ctx.tokenizer.peek() is None:
            return BAD_PARSE
        elif ctx.tokenizer.peek().type == self.name:
            t = ctx.tokenizer.next()
            return (t.value, t.info)
        return BAD_PARSE
    def __str__(self):
        return '"%s"' % self.name

class Repeat:
    def __init__(self, rule, item, min=0):
        self.rule = rule
        self.item = item
        self.min = min
    def parse(self, ctx):
        results = []
        item = self.item.parse(ctx)
        while item is not BAD_PARSE:
            results.append(item)
            item = self.item.parse(ctx)
        return unzip(results) if len(results) >= self.min else BAD_PARSE
    def __str__(self):
        return 'rep(%s)' % self.item

class Seq:
    def __init__(self, rule, items):
        self.rule = rule
        self.items = items
    def parse(self, ctx):
        items = []
        pos = ctx.tokenizer.pos
        for item in self.items:
            r = item.parse(ctx)
            if r is BAD_PARSE:
                ctx.tokenizer.pos = pos
                return BAD_PARSE
            items.append(r)
        return unzip(items)
    def __str__(self):
        return 'seq(%s)' % ','.join(map(str, self.items))

class Alt:
    def __init__(self, rule, items):
        self.rule = rule
        self.items = items
    def parse(self, ctx):
        for item in self.items:
            r = item.parse(ctx)
            if r is not BAD_PARSE:
                return r
        return BAD_PARSE
    def __str__(self):
        return 'alt(%s)' % ','.join(map(str, self.items))

class Opt:
    def __init__(self, rule, item):
        self.rule = rule
        self.item = item
    def parse(self, ctx):
        result = self.item.parse(ctx)
        return (None, None) if result is BAD_PARSE else result
    def __str__(self):
        return 'opt(%s)' % self.item

class FnWrapper:
    def __init__(self, rule, prod, fn):
        if not isinstance(prod, Seq):
            prod = Seq(rule, [prod])
        self.rule = rule
        self.prod = prod
        self.fn = fn
    def parse(self, ctx):
        result = self.prod.parse(ctx)
        if result is not BAD_PARSE:
            result, info = result
            return (self.fn(ParseResult(result, info)), info)
        return BAD_PARSE
    def __str__(self):
        return str(self.prod)

# Mini parser for our grammar specification language (basically EBNF)

def parse_repeat(rule, tokenizer, repeated):
    if tokenizer.accept('STAR'):
        return Repeat(rule, repeated)
    elif tokenizer.accept('PLUS'):
        return Repeat(rule, repeated, min=1)
    return repeated

def parse_rule_atom(rule, tokenizer):
    if tokenizer.accept('LPAREN'):
        r = parse_rule_expr(rule, tokenizer)
        tokenizer.expect('RPAREN')
        r = parse_repeat(rule, tokenizer, r)
    elif tokenizer.accept('LBRACKET'):
        r = Opt(rule, parse_rule_expr(rule, tokenizer))
        tokenizer.expect('RBRACKET')
    else:
        t = tokenizer.accept('IDENT')
        if t:
            r = parse_repeat(rule, tokenizer, String(rule, t.value))
        else:
            raise RuntimeError('bad token: %s' % tokenizer.peek())
    return r

def parse_rule_seq(rule, tokenizer):
    r = []
    tok = tokenizer.peek()
    while tok and tok.type != 'RBRACKET' and tok.type != 'RPAREN' and tok.type != 'PIPE':
        r.append(parse_rule_atom(rule, tokenizer))
        tok = tokenizer.peek()
    if len(r) > 1:
        return Seq(rule, r)
    return r[0] if r else None

def parse_rule_expr(rule, tokenizer):
    r = [parse_rule_seq(rule, tokenizer)]
    while tokenizer.accept('PIPE'):
        r.append(parse_rule_seq(rule, tokenizer))
    if len(r) > 1:
        return Alt(rule, r)
    return r[0]

# ...And a mini lexer too

rule_tokens = {
    'IDENT': '[a-zA-Z_]+',
    'LBRACKET': '\[',
    'LPAREN': '\(',
    'PIPE': '\|',
    'RBRACKET': '\]',
    'RPAREN': '\)',
    'STAR': '\*',
    'PLUS': '\+',
    'WHITESPACE': ' ',
}
skip = {'WHITESPACE'}

# Utility functions

def rule_fn(rule_table, rule, prod):
    def wrapper(fn):
        rule_table.append((rule, (prod, fn)))
        return fn
    return wrapper

class Parser:
    def __init__(self, rule_table, start):
        self.tokenizer = liblex.Tokenizer(rule_tokens, skip)
        self.fn_table = {}
        for [rule, *prods] in rule_table:
            for prod in prods:
                fn = None
                if isinstance(prod, tuple):
                    prod, fn = prod
                self.create_rule(rule, prod, fn)
        self.start = start

    def create_rule(self, rule, prod, fn):
        prod = parse_rule_expr(rule, self.tokenizer.input(prod))
        prod = FnWrapper(rule, prod, fn) if fn else prod
        if rule not in self.fn_table:
            self.fn_table[rule] = Alt(rule, [])
        self.fn_table[rule].items.append(prod)

    def parse(self, tokenizer):
        prod = self.fn_table[self.start]
        result = prod.parse(Context(self.fn_table, tokenizer))
        if result is BAD_PARSE:
            raise RuntimeError('bad parse near token %s' % tokenizer.peek())
        elif tokenizer.peek() is not None:
            raise RuntimeError('parser did not consume entire input, near token %s' %
                tokenizer.peek())
        result, info = result
        return result
