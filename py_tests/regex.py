import re

import re_test_cases

for r in re_test_cases.regexes:
    reg = re.compile(r)
    print([bool(x is not None) for x in map(reg.match, re_test_cases.inputs)])
