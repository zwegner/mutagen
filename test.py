#!/usr/bin/env python3
import os
import subprocess
import sys
import time

root_dir = sys.path[0]
mutagen = '%s/bootstrap/parse.py' % root_dir
python = 'python3'

passes = fails = 0

test_set = set(sys.argv[1:])

# Mutagen tests: these should just not throw any errors
mg_tests = ['builtins', 'core']
mg_test_dir = 'mg_tests'

# Python tests: python/mutagen should produce the same output in each case.
py_tests = ['regex', 'lex']
py_test_dir = 'py_tests'

start = time.time()

# Run mutagen tests
for test in mg_tests:
    if test_set and test not in test_set:
        continue
    try:
        path = '%s.mg' % (test)
        subprocess.check_output([mutagen, path], cwd=mg_test_dir)
        passes += 1
    except Exception:
        fails += 1
        print('TEST %s FAILED!' % test)

for test in py_tests:
    if test_set and test not in test_set:
        continue
    # Run python version
    path = '%s.py' % (test)
    py_output = subprocess.check_output([python, path], cwd=py_test_dir)

    # Run mutagen version
    path = '%s.mg' % (test)
    mg_output = subprocess.check_output([mutagen, path], cwd=py_test_dir)

    if py_output != mg_output:
        fails += 1
        print('TEST %s FAILED!' % test)
        print('python output:\n%s' % py_output.decode('utf-8'))
        print('mutagen output:\n%s' % mg_output.decode('utf-8'))
    else:
        passes += 1

end = time.time()
print('%s/%s tests passed. Time: %.2fs' % (passes, passes + fails, end - start))
sys.exit(fails != 0)
