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
            match = m.match(s)
            if match[0]:
                r = 1
        if self.inv:
            r = not r
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
        r = self.left.match(s)
        if r[0]:
            l = r[1]
            r = self.right.match(slice(s, r[1], len(s)))
            r = [r[0], l+r[1]]
        r

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
    def __init__(r):
        make(['regex', r])

    def match(self, s):
        l = 0
        r = self.regex.match(s)
        while r[0] != 0:
            l = l + r[1]
            r = self.regex.match(slice(s, l, len(s)))
        0
        [1, l]

# Null regex: matches nothing.
class MatchNull:
    def match(self, s):
        [1, 0]

def parse(string):
    result = MatchNull()
    c = 0
    while c < len(string):
        if string[c] == '\\':
            c = c + 1
            current_match = MatchChar(string[c])
        elif string[c] == '.':
            current_match = MatchAny()
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
            current_match = MatchOpt(inv, opts)
        else:
            current_match = MatchChar(string[c])
        result = MatchSeq(result, current_match)
        c = c + 1
    result
