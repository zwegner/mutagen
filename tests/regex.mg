import re

import re_test_cases

for r in re_test_cases.regexes:
    reg = re.parse(r)
    print(map(lambda(x){return x[0]}, map(reg.match, re_test_cases.inputs)))

