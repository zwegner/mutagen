from langdeps import *

# Test string operations
print('a' + 'b' + 'c')
print('abc' * 0)
print('abc' * 1)
print('abc' * 20)

# Test string escape sequences
for escape in ['\x00', '\x20', '\x40', '\x80', '\xFF',
        '\n', '\t', '\b', '\'', '\\']:
    print(escape)

# Test string methods
for test_case in ['a', 'abc', 'ABC', 'abcXYZ', '123']:
    for method in [str.islower, str.lower, str.isupper, str.upper]:
        print(method(test_case))

# Test encoding--use list() to make sure it prints as a list of integers
print(list('ABCabc!@#$%^&*()'.encode('ascii')))
print(list('œ∑´®†¥¨ˆøπåß∂ƒ©˙∆˚¬Ω≈ç√∫˜µ'.encode('utf-8')))
assert_call_fails(str.encode, '\x80', 'ascii') # test bad ascii char

# Test various type stuff
print(type)
print(type(0))
print(type(type(0)))
print(type('abc'))
print(type(None))

# Test list operations
print([0] + [1])
print([0] * 0)
print([0] * 1)
print([0] * 20)
print([0, 1, 2] * 20)
