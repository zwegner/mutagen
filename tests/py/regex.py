import re

from langdeps import *

regexes = [
    '',
    '.*',
    '.+',
    '[A-Z]',
    '[\\[]',
    '[]]',
    '[^[]',
    '[^[ab]',
    '[^a-b]',
    '[^b-z]',
    '[a-b]*',
    '[a-z]',
    '[a-z]b',
    '[b-c]+',
    '[b-z]',
    '][a]',
    'a?a?b',
    'aa|b',
    '(a?)*b',
    'aab|bbb|(aa*)',
    '(a|b)*',
    'asdf|',
]
inputs = [
    '',
    '[',
    ']',
    ']a',
    'aa',
    'aab',
    'aaaaaab',
    'aabaaab',
    'aa',
    'aa',
    'aa',
    'b',
    'ba',
    'bab',
    'bba',
    'bbb',
    'bca',
]

for r in regexes:
    reg = re.compile(r)
    print([bool(reg.match(i, 0) != None) for i in inputs])

bad_regexes = [
    '*',
    '(asdf',
    'asdf)',
]

for r in bad_regexes:
    assert_call_fails(re.compile, r)
