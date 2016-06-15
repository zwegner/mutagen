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

class Save(index):
    def __repr__(self):
        return 'save {}'.format(self.index)

class Done:
    def __repr__(self):
        return 'done'

class Match(save_points, groups):
    def span(self, group):
        return self.save_points[group * 2:group * 2 + 2]
    def get_last_group(self):
        group = None
        for i in range(3, len(self.save_points), 2):
            if self.save_points[i] != None and self.groups[i // 2] != None:
                group = self.groups[i // 2]
        return group

class Pattern(states, groups):
    def match(self, s, start):
        def add_state(ss, sps, s, sp):
            if s not in sps:
                ss = ss + [s]
                sps = sps <- [s] = sp
            return [ss, sps]
        save = [None] * len(self.groups) * 2
        states = [0]
        save_points = {0: save}
        result = None
        for i in range(start, len(s) + 1):
            if not states:
                break
            c = None
            if i < len(s):
                c = s[i]
            new_states = []
            new_save_points = {}
            n = 0
            while n < len(states):
                state_id = states[n]
                save = save_points[state_id]
                state = self.states[state_id]
                if isinstance(state, Jump):
                    [states, save_points] = add_state(states, save_points,
                        state_id + state.offset, save)
                elif isinstance(state, Split):
                    [states, save_points] = add_state(states, save_points,
                        state_id + state.offset1, save)
                    [states, save_points] = add_state(states, save_points,
                        state_id + state.offset2, save)
                elif isinstance(state, Save):
                    save = save <- [state.index] = i
                    [states, save_points] = add_state(states, save_points,
                        state_id + 1, save)
                elif isinstance(state, Done):
                    result = Match(save, self.groups)
                elif c != None and state.matches(c):
                    [new_states, new_save_points] = add_state(new_states,
                        new_save_points, state_id + 1, save)
                n = n + 1
            states = new_states
            save_points = new_save_points

        return result

    def __repr__(self):
        lines = []
        for state in self.states:
            if (isinstance(state, Jump) or isinstance(state, Split) or
                isinstance(state, Done) or isinstance(state, Save)):
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

def parse_group(parse_alt, string, c, groups):
    states = []
    while c < len(string) and string[c] not in ['|', ')']:
        if string[c] == '(':
            c = c + 1
            group_name = None
            if c < len(string) and string[c] == '?':
                c = c + 1
                if c + 1 < len(string) and string[c] == 'P' and string[c + 1] == '<':
                    c = c + 2
                    group_name = ''
                    while c < len(string) and string[c] != '>':
                        group_name = group_name + string[c]
                        c = c + 1
                    c = c + 1
                else:
                    error('regex parsing error at char {}: "{}"'.format(c, string[c]))
            group_id = len(groups)
            groups = groups + [group_name]

            [new_states, c, groups] = parse_alt(string, c, groups)
            if string[c] != ')':
                error('regex parsing error at char {}: "{}"'.format(c, string[c]))
            c = c + 1

            new_states = [Save(group_id * 2)] + new_states + [Save(group_id * 2 + 1)]
        else:
            [new_states, c] = parse_item(string, c)

        # Parse repeaters
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
    return [states, c, groups]

@fixed_point
def parse_alt(parse_alt, string, c, groups):
    [states, c, groups] = parse_group(parse_alt, string, c, groups)

    # Parse the alternator character--use right recursion for simplicity
    if c < len(string) and string[c] == '|':
        [other_states, c, groups] = parse_alt(string, c + 1, groups)
        # Try to match the first item, or the second
        states = ([Split(1, len(states) + 2)] + states +
                [Jump(len(other_states) + 1)] + other_states)

    return [states, c, groups]

def compile(string):
    [states, c, groups] = parse_alt(string, 0, [None])
    if c < len(string):
        error('regex parsing error at char {}: "{}"'.format(c, string[c]))
    states = [Save(0)] + states + [Save(1), Done()]
    return Pattern(states, groups)
