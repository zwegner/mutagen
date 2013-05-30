import re

# Cheap-ass cross-language import
with open('re_test_cases.mg') as f:
    exec(f.read())

for r in regexes:
    reg = re.compile(r)
    print([bool(x is not None) for x in map(reg.match, inputs)])
