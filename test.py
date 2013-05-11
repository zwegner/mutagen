#!/usr/bin/env python3
import os
import subprocess
import sys

root_dir = sys.path[0]
mutagen = '%s/bootstrap/parse.py' % root_dir
python = 'python3'

passes = fails = 0

py_tests = ['regex']
py_test_dir = 'tests'

for test in py_tests:
    # Run python version
    path = '%s.py' % (test)
    py_output = subprocess.check_output([python, path], cwd=py_test_dir)

    # Run mutagen version
    path = '%s.mg' % (test)
    mg_output = subprocess.check_output([mutagen, path], cwd=py_test_dir)

    if py_output != mg_output:
        fails += 1
        print('TEST %s FAILED!' % test)
        print('python output:\n%s' % py_output)
        print('mutagen output:\n%s' % mg_output)
    else:
        passes += 1

print('%s/%s tests passed.' % (passes, passes + fails))
sys.exit(fails != 0)
