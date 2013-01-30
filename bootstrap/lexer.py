tokens = {
    '=': 'equals',
    '(': 'lparen',
    ')': 'rparen',
    '[': 'lbracket',
    ']': 'rbracket',
    '{': 'lbrace',
    '}': 'rbrace',
    ',': 'comma',
    '.': 'period',
    ';': 'semicolon',
    '\n': 'newline',
}
keywords = {
    'import',
}

class Lexer:
    def __init__(self, input_file):
        self.input = input_file
        self.saved_char = None

    def get_char(self):
        if self.saved_char:
            c = self.saved_char
            self.saved_char = None
            return c
        return self.input.read(1)

    def get_token_inner(self):
        while True:
            c = self.get_char()
            # EOF
            if c == '':
                return None
            # Comments
            elif c == '#':
                while True:
                    c = self.get_char()
                    if not c or c == '\n':
                        break
                return ('newline', '\n')
            # Strings
            elif c == '\'':
                string = ''
                while True:
                    c = self.get_char()
                    # TODO: escapes
                    if c == '\'':
                        break
                    string += c
                return ('string', string)
            # Identifiers
            elif c.isalpha():
                identifier = c
                while True:
                    c = self.get_char()
                    if not c.isalpha() and not c.isdigit() and not c == '_':
                        self.saved_char = c
                        break
                    identifier += c
                if identifier in keywords:
                    return (identifier, identifier)
                return ('ident', identifier)
            elif c in tokens:
                return (tokens[c], c)

    def get_token(self):
        t = self.get_token_inner()
        if not t:
            return ('EOF', '')
        return t
