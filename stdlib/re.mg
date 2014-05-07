# char: match a single character
class MatchChar(char):
    def matches(self, c):
        return c == self.char
    def __repr__(self):
        return self.char

# Any: match any single character
class MatchAny:
    def matches(self, c):
        return True
    def __repr__(self):
        return '.'

# Opt: equivalent to a [A-Za-z] group in a regex, matches
# a set of ranges/single characters, or the inverse of this set
class MatchOpt(inv, options):
    def matches(self, c):
        r = any([m.matches(c) for m in self.options])
        if self.inv:
            r = not r
        return r
    def __repr__(self):
        inv = ''
        if self.inv:
            inv = '^'
        return '[{}{}]'.format(inv, ''.join(map(str, self.options)))

# Range: match a character that's in a range
class MatchRange(low, high):
    def matches(self, c):
        return c >= self.low and c <= self.high
    def __repr__(self):
        return '{}-{}'.format(self.low, self.high)

# Control flow classes: used for jumping to certain places in the NFA.
class Jump(offset):
    def __repr__(self):
        return 'jump {}'.format(self.offset)

class Split(offset1, offset2):
    def __repr__(self):
        return 'split {}, {}'.format(self.offset1, self.offset2)

class Done:
    def __repr__(self):
        return 'done'

class Pattern(states):
    def match(self, s):
        def add_state(l, s):
            if s not in l:
                l = l + [s]
            return l
        states = [0]
        result = None
        for i in range(len(s) + 1):
            if not states:
                break
            c = None
            if i < len(s):
                c = s[i]
            new_states = []
            n = 0
            while n < len(states):
                state_id = states[n]
                state = self.states[state_id]
                if isinstance(state, Jump):
                    states = add_state(states, state_id + state.offset)
                elif isinstance(state, Split):
                    states = add_state(states, state_id + state.offset1)
                    states = add_state(states, state_id + state.offset2)
                elif isinstance(state, Done):
                    result = [True, i]
                elif c != None and state.matches(c):
                    new_states = add_state(new_states, state_id + 1)
                n = n + 1
            states = new_states

        return result

    def __repr__(self):
        lines = []
        for state in self.states:
            if (isinstance(state, Jump) or isinstance(state, Split) or
                isinstance(state, Done)):
                lines = lines + [str(state)]
            else:
                lines = lines + ['match {}'.format(state)]
        return '\n'.join(lines)

def parse_item(string, c):
    # Backslash: escape next char
    # XXX handle escape sequences properly
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
    # Special characters
    elif string[c] in ['(', ')', '*', '+', '|', '?']:
        error('regex parsing error at char {}: "{}"'.format(c, string[c]))
    # Match a literal character
    else:
        item = MatchChar(string[c])
    return [[item], c + 1]

def parse_group(parse_alt, string, c):
    states = []
    while c < len(string) and string[c] != ')':
        if string[c] == '(':
            [new_states, c] = parse_alt(string, c + 1)
            if string[c] != ')':
                error('regex parsing error at char {}: "{}"'.format(c, string[c]))
            c = c + 1
        elif string[c] == '|':
            return [states, c]
        else:
            [new_states, c] = parse_item(string, c)

        # Parse repeaters/connectors
        if c < len(string):
            if string[c] == '*':
                # Try to either match the next item and do it again, or
                # move ahead past it
                new_states = [Split(1, len(new_states) + 2)] + new_states + [
                    Jump(-len(new_states) - 1)]
                c = c + 1
            elif string[c] == '+':
                # Match the next item, then try it again or not
                new_states = new_states + [Split(-len(new_states), 1)]
                c = c + 1
            elif string[c] == '?':
                # Match the next item or skip it
                new_states = [Split(1, len(new_states) + 1)] + new_states
                c = c + 1
        states = states + new_states
    return [states, c]

@fixed_point
def parse_alt(parse_alt, string, c):
    [states, c] = parse_group(parse_alt, string, c)

    # Parse the alternator character--use right recursion for simplicity
    if c < len(string) and string[c] == '|':
        [other_states, c] = parse_alt(string, c + 1)
        # Try to match the first item, or the second
        states = ([Split(1, len(states) + 2)] + states +
                [Jump(len(other_states) + 1)] + other_states)

    return [states, c]

def compile(string):
    [states, c] = parse_alt(string, 0)
    if c < len(string):
        error('regex parsing error at char {}: "{}"'.format(c, string[c]))
    return Pattern(states + [Done()])
