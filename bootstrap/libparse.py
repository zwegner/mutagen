import liblex

# Classes to represent grammar structure

class String:
    def __init__(self, name):
        self.name = name
    def parse(self, tokenizer, fn_table):
        if self.name in fn_table:
            return fn_table[self.name].parse(tokenizer, fn_table)
        elif tokenizer.peek() is None:
            return None
        elif tokenizer.peek().type == self.name:
            t = tokenizer.next()
            return t.value
    def __str__(self):
        return '"%s"' % self.name

class Repeat:
    def __init__(self, item):
        self.item = item
    def parse(self, tokenizer, fn_table):
        results = []
        item = self.item.parse(tokenizer, fn_table)
        while item != None:
            results.append(item)
            item = self.item.parse(tokenizer, fn_table)
        return results
    def __str__(self):
        return 'rep(%s)' % self.item

class Seq:
    def __init__(self, items):
        self.items = items
    def parse(self, tokenizer, fn_table):
        items = []
        for i, item in enumerate(self.items):
            r = item.parse(tokenizer, fn_table)
            if r is None:
                # Instead of proper error messages, explode randomly
                assert i == 0
                return None
            items.append(r)
        return items[0] if len(items) == 1 else items
    def __str__(self):
        return 'seq(%s)' % ','.join(map(str, self.items))

class Alt:
    def __init__(self, items):
        self.items = items
    def parse(self, tokenizer, fn_table):
        for item in self.items:
            r = item.parse(tokenizer, fn_table)
            if r is not None:
                return r
        return None
    def __str__(self):
        return 'alt(%s)' % ','.join(map(str, self.items))

class Opt:
    def __init__(self, item):
        self.item = item
    def parse(self, tokenizer, fn_table):
        return self.item.parse(tokenizer, fn_table) or ''
    def __str__(self):
        return 'opt(%s)' % self.item

class FnWrapper:
    def __init__(self, prod, fn):
        self.prod = prod
        self.fn = fn
    def parse(self, tokenizer, fn_table):
        result = self.prod.parse(tokenizer, fn_table)
        if result:
            return self.fn(result)
        return None

# Mini parser for our grammar specification language (basically EBNF)

def parse_rule_atom(tokenizer):
    if tokenizer.accept('LPAREN'):
        r = parse_rule_expr(tokenizer)
        tokenizer.expect('RPAREN')
        if tokenizer.accept('STAR'):
            r = Repeat(r)
    elif tokenizer.accept('LBRACKET'):
        r = Opt(parse_rule_expr(tokenizer))
        tokenizer.expect('RBRACKET')
    else:
        t = tokenizer.accept('IDENT')
        if t:
            r = String(t.value)
            if tokenizer.accept('STAR'):
                r = Repeat(r)
        else:
            raise RuntimeError('bad tok: %s' % (tokenizer.peek(),))
    return r

def parse_rule_seq(tokenizer):
    r = []
    tok = tokenizer.peek()
    while tok and tok.type != 'RBRACKET' and tok.type != 'RPAREN' and tok.type != 'PIPE':
        r.append(parse_rule_atom(tokenizer))
        tok = tokenizer.peek()
    if len(r) > 1:
        return Seq(r)
    return r[0] if r else None

def parse_rule_expr(tokenizer):
    r = [parse_rule_seq(tokenizer)]
    while tokenizer.accept('PIPE'):
        r.append(parse_rule_seq(tokenizer))
    if len(r) > 1:
        return Alt(r)
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
        rule_table[rule] = (prod, fn)
        return fn
    return wrapper

class Parser:
    def __init__(self, rule_table, start):
        self.tokenizer = liblex.Tokenizer(rule_tokens, skip)
        self.fn_table = {}
        for k, v in sorted(rule_table.items()):
            fn = None
            if isinstance(v, tuple):
                v, fn = v
            self.create_rule(k, v, fn)
        self.start = start

    def create_rule(self, rule, prod, fn):
        self.tokenizer.input(prod)
        prod = parse_rule_expr(self.tokenizer)
        prod = FnWrapper(prod, fn) if fn else prod
        if rule not in self.fn_table:
            self.fn_table[rule] = Alt([])
        self.fn_table[rule].items.append(prod)

    def parse(self, tokenizer):
        prod = self.fn_table[self.start]
        return prod.parse(tokenizer, self.fn_table)
