# Test string escape sequences
for escape in ['\x00', '\x20', '\x40', '\x80', '\xFF',
        '\n', '\t', '\b', '\'', '\\']:
    print(escape)

# Test various type stuff
print(type)
print(type(0))
print(type(type(0)))
print(type('abc'))
print(type(None))
