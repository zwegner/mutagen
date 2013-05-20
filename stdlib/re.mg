# char: match a single character
class MatchChar:
    def __init__(c):
        make(['char', c])

    def match(self, s):
        if len(s):
            [s[0] == self.char, 1]
        else:
            [0, 0]

# Any: match any single character
class MatchAny:
    def match(self, s):
        match = len(s) > 0
        [match, match]

# Opt: equivalent to a [A-Za-z] group in a regex, matches
# a set of ranges/single characters, or the inverse of this set
class MatchOpt:
    def __init__(inv, matches):
        make(['inv', inv], ['matches', matches])

    def match(self, s):
        r = 0
        for m in self.matches:
            [match, l] = m.match(s)
            if match:
                r = 1
        if self.inv:
            r = not r and len(s) > 0
        [r, 1]

# Range: match a character that's in a range
class MatchRange:
    def __init__(low, high):
        make(['low', low], ['high', high])

    def match(self, s):
        if len(s):
            [s[0] >= self.low and s[0] <= self.high, 1]
        else:
            [0, 0]

# Sequence: take two regexes and match them both in order.
class MatchSeq:
    def __init__(a, s):
        make(['left', a], ['right', s])

    def match(self, s):
        [r, l] = self.left.match(s)
        if r:
            [r, i] = self.right.match(slice(s, l, len(s)))
            l = l + i
        [r, l]

# Alternation: match either of two regexes
class MatchAlt:
    def __init__(a, s):
        make(['left', a], ['right', s])

    def match(self, s):
        r = self.left.match(s)
        if not r[0]:
            r = self.right.match(s)
        r

# Repetition: match as many times as possible
class MatchRep:
    def __init__(r, min, max):
        make(['regex', r], ['min', min])

    def match(self, s):
        l = 0
        count = 0
        [r, i] = self.regex.match(s)
        while r != 0:
            l = l + i
            count = count + 1
            [r, i] = self.regex.match(slice(s, l, len(s)))
        [count >= self.min, l]

# Null regex: matches nothing.
class MatchNull:
    def match(self, s):
        [1, 0]

def parse_item(string, c):
    # Backslash: escape next char
    if string[c] == '\\':
        c = c + 1
        item = MatchChar(string[c])
    # Period: any character
    elif string[c] == '.':
        item = MatchAny()
    # Parse option
    elif string[c] == '[':
        c = c + 1
        opts = []
        if string[c] == '^':
            c = c + 1
            inv = 1
        else:
            inv = 0
        # Special case: []] includes a literal ']' char
        if string[c] == ']':
            c = c + 1
            opts = opts + [MatchChar(']')]
        # Keep parsing ranges/single chars until we hit
        while string[c] != ']':
            low = string[c]
            if string[c+1] == '-':
                high = string[c+2]
                c = c + 2
                opts = opts + [MatchRange(low, high)]
            else:
                opts = opts + [MatchChar(low)]
            c = c + 1
        item = MatchOpt(inv, opts)
    # Parentheses: group a number of items together
    elif string[c] == '(':
        c = c + 1
        [item, c] = parse_group(string, c)
        if string[c] != ')':
            error
    # Special characters
    elif string[c] == '*' or string[c] == '+' or string[c] == '|' or string[c] == '?':
        error
    # Match a literal character
    else:
        item = MatchChar(string[c])
    [item, c + 1]

def parse_group(string, c):
    result = Nil
    while c < len(string) and string[c] != ')':
        old_c = c
        [item, c] = parse_item(string, c)
        # Parse repeaters/connectors
        if c < len(string):
            if string[c] == '*':
                c = c + 1
                item = MatchRep(item, 0)
            elif string[c] == '+':
                c = c + 1
                item = MatchRep(item, 1)
            elif string[c] == '?':
                c = c + 1
                item = MatchAlt(item, MatchNull())
            else:
                while c < len(string) and string[c] == '|':
                    c = c + 1
                    if c >= len(string):
                        error
                    [alt, c] = parse_item(string, c)
                    item = MatchAlt(item, alt)
        if result == Nil:
            result = item
        else:
            result = MatchSeq(result, item)
    [result, c]

def parse(string):
    result_c = parse_group(string, 0)
    result = result_c[0]
    c = result_c[1]
    if c < len(string):
        error
    result
