# char: match a single character
class MatchChar(char):
    def match(self, s):
        if len(s):
            return [s[0] == self.char, 1]
        return [False, 0]

# Any: match any single character
class MatchAny:
    def match(self, s):
        match = len(s) > 0
        if match {c = 1;} else {c = 0;}
        return [match, c]

# Opt: equivalent to a [A-Za-z] group in a regex, matches
# a set of ranges/single characters, or the inverse of this set
class MatchOpt(inv, matches):
    def match(self, s):
        r = False
        for m in self.matches:
            [match, l] = m.match(s)
            if match:
                r = True
                break
        if self.inv:
            r = not r and len(s) > 0
        return [r, 1]

# Range: match a character that's in a range
class MatchRange(low, high):
    def match(self, s):
        if len(s) and s[0] >= self.low and s[0] <= self.high:
            return [True, 1]
        return [False, 0]

# Sequence: take two regexes and match them both in order.
class MatchSeq(left, right):
    def match(self, s):
        [r, l] = self.left.match(s)
        if r:
            [r, i] = self.right.match(slice(s, l, None))
            l = l + i
        return [r, l]

# Alternation: match either of two regexes
class MatchAlt(left, right):
    def match(self, s):
        r = self.left.match(s)
        if not r[0]:
            r = self.right.match(s)
        return r

# Repetition: match as many times as possible
class MatchRep(regex, min):
    def match(self, s):
        l = 0
        count = 0
        [r, i] = self.regex.match(s)
        while r:
            l = l + i
            count = count + 1
            [r, i] = self.regex.match(slice(s, l, None))
        return [count >= self.min, l]

# Null regex: matches nothing.
class MatchNull:
    def match(self, s):
        return [True, 0]

# XXX recursion
def parse_item(pg, string, c):
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
            inv = True
        else:
            inv = False
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
        [item, c] = pg(pg, string, c)
        if string[c] != ')':
            error('regex parsing error')
    # Special characters
    elif string[c] == '*' or string[c] == '+' or string[c] == '|' or string[c] == '?':
        error('regex parsing error')
    # Match a literal character
    else:
        item = MatchChar(string[c])
    return [item, c + 1]

# XXX recursion
def parse_group(pg, string, c):
    result = None
    while c < len(string) and string[c] != ')':
        [item, c] = parse_item(pg, string, c)
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
                        error('regex parsing error')
                    [alt, c] = parse_item(pg, string, c)
                    item = MatchAlt(item, alt)
        if result == None:
            result = item
        else:
            result = MatchSeq(result, item)
    return [result, c]

def parse(string):
    [result, c] = parse_group(parse_group, string, 0)
    if c < len(string):
        error('regex parsing error')
    return result
