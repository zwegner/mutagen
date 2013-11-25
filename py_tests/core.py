# Test string escape sequences
for escape in [
    '\x00',
    '\x20',
    '\x40',
    '\x80',
    '\xFF',
    '\n',
    '\t',
    '\b',
    '\'',
    '\\'
    ]:
    print(escape)
