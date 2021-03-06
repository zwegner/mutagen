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
print([0] * -5)
print([0] * 0)
print([0] * 1)
print([0] * 20)
print([0, 1, 2] * 20)

# Test list and dict comprehensions, making sure scoping is handled properly
x = 99
y = True
print([x * y for y in range(4)])
print({y: x for y in range(4)})
print([[x * y for y in range(4)] for x in range(4)])
print([{x * y: x + y for y in range(4)} for x in range(4)])
print({xxx: xxx for xxx in range(4)})
def with_new_scope():
    print(x) # Make sure to access global scope... wacky bug
    print([zzz * 4 + yyy for yyy in range(4) for zzz in range(4)])
    print({yyy: yyy for yyy in range(4)})
    print({zzz * 4 + yyy: yyy for yyy in range(4) for zzz in range(4)})
with_new_scope()
print(x)
print(y)
