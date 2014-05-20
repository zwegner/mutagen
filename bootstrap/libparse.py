import liblex

class Context:
    def __init__(self, fn_table):
        self.fn_table = fn_table
        self.check_prods = set()

# Dummy sentinel object
BAD_PARSE = object()

# Classes to represent grammar structure

class String:
    def __init__(self, rule, name):
        self.rule = rule
        self.name = name
    def parse(self, tokenizer, fn_table):
        if self.name in fn_table:
            return fn_table[self.name].parse(tokenizer, fn_table)
        elif tokenizer.peek() is None:
            return BAD_PARSE
        elif tokenizer.peek().type == self.name:
            t = tokenizer.next()
            return t.value
        return BAD_PARSE
    def check_first_token(self, ctx):
        if self.name in ctx.fn_table:
            return ctx.fn_table[self.name].check_first_token(ctx)
        # XXX check whether token type exists
        return {self.name}
    def __str__(self):
        return '"%s"' % self.name

class Repeat:
    def __init__(self, rule, item):
        self.rule = rule
        self.item = item
    def parse(self, tokenizer, fn_table):
        results = []
        item = self.item.parse(tokenizer, fn_table)
        while item is not BAD_PARSE:
            results.append(item)
            item = self.item.parse(tokenizer, fn_table)
        return results
    def check_first_token(self, ctx):
        return self.item.check_first_token(ctx)
    def __str__(self):
        return 'rep(%s)' % self.item

class Seq:
    def __init__(self, rule, items):
        self.rule = rule
        self.items = items
    def parse(self, tokenizer, fn_table):
        items = []
        for i, item in enumerate(self.items):
            r = item.parse(tokenizer, fn_table)
            if r is BAD_PARSE:
                if i > 0:
                    raise RuntimeError('parsing error: got %s while looking for %s '
                        'in rule %s: %s' % (tokenizer.peek(), item, self.rule, self))
                return BAD_PARSE
            items.append(r)
        return items[0] if len(items) == 1 else items
    def check_first_token(self, ctx):
        # Check that all later symbols are unambiguously parseable, but only
        # if we're not being called recursively
        if self.items[0] not in ctx.check_prods:
            ctx.check_prods.add(self.items[0])
            [item.check_first_token(ctx) for item in self.items[1:]]
            ctx.check_prods.remove(self.items[0])
        return self.items[0].check_first_token(ctx)
    def __str__(self):
        return 'seq(%s)' % ','.join(map(str, self.items))

class Alt:
    def __init__(self, rule, items):
        self.rule = rule
        self.items = items
    def parse(self, tokenizer, fn_table):
        for item in self.items:
            r = item.parse(tokenizer, fn_table)
            if r is not BAD_PARSE:
                return r
        return BAD_PARSE
    def check_first_token(self, ctx):
        firsts = [item.check_first_token(ctx) for item in self.items]
        all_firsts = set(item for f in firsts for item in f)
        if len(all_firsts) != sum(len(f) for f in firsts):
            raise RuntimeError('ambiguous parser specification near: %s' % self)
        return all_firsts
    def __str__(self):
        return 'alt(%s)' % ','.join(map(str, self.items))

class Opt:
    def __init__(self, rule, item):
        self.rule = rule
        self.item = item
    def parse(self, tokenizer, fn_table):
        result = self.item.parse(tokenizer, fn_table)
        return [] if result is BAD_PARSE else result
    def check_first_token(self, ctx):
        return self.item.check_first_token(ctx)
    def __str__(self):
        return 'opt(%s)' % self.item

class FnWrapper:
    def __init__(self, rule, prod, fn):
        self.rule = rule
        self.prod = prod
        self.fn = fn
    def parse(self, tokenizer, fn_table):
        result = self.prod.parse(tokenizer, fn_table)
        if result is not BAD_PARSE:
            return self.fn(result)
        return BAD_PARSE
    def check_first_token(self, ctx):
        return self.prod.check_first_token(ctx)
    def __str__(self):
        return str(self.prod)

# Mini parser for our grammar specification language (basically EBNF)

def parse_rule_atom(rule, tokenizer):
    if tokenizer.accept('LPAREN'):
        r = parse_rule_expr(rule, tokenizer)
        tokenizer.expect('RPAREN')
        if tokenizer.accept('STAR'):
            r = Repeat(rule, r)
    elif tokenizer.accept('LBRACKET'):
        r = Opt(rule, parse_rule_expr(rule, tokenizer))
        tokenizer.expect('RBRACKET')
    else:
        t = tokenizer.accept('IDENT')
        if t:
            r = String(rule, t.value)
            if tokenizer.accept('STAR'):
                r = Repeat(rule, r)
        else:
            raise RuntimeError('bad tok: %s' % (tokenizer.peek(),))
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
        for k, v in rule_table:
            fn = None
            if isinstance(v, tuple):
                v, fn = v
            self.create_rule(k, v, fn)
        self.start = start

    def create_rule(self, rule, prod, fn):
        self.tokenizer.input(prod)
        prod = parse_rule_expr(rule, self.tokenizer)
        prod = FnWrapper(rule, prod, fn) if fn else prod
        if rule not in self.fn_table:
            self.fn_table[rule] = Alt(rule, [])
        self.fn_table[rule].items.append(prod)
        self.fn_table[rule].check_first_token(Context(self.fn_table))

    def parse(self, tokenizer):
        prod = self.fn_table[self.start]
        return prod.parse(tokenizer, self.fn_table)
