#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile
import time

root_dir = os.path.dirname(sys.path[0])
mutagen = '%s/bootstrap/parse.py' % root_dir
python = 'python3'

passes = fails = 0

test_set = set(sys.argv[1:])

# Mutagen tests: these should just not throw any errors
mg_tests = ['builtins', 'core']
mg_test_dir = 'tests/mg/'

# Python tests: python/mutagen should produce the same output in each case.
py_tests = ['core', 'regex', 'lex']
py_test_dir = 'tests/py'

raw_tests = ['asm']
raw_test_dir = 'tests/raw'

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
        print('MUTAGEN TEST %s FAILED!' % test)

for test in py_tests:
    if test_set and test not in test_set:
        continue

    outputs = []
    for [interpreter, ext] in [[python, 'py'], [mutagen, 'mg']]:
        path = '%s.%s' % (test, ext)
        try:
            output = subprocess.check_output([interpreter, path], cwd=py_test_dir)
        except subprocess.CalledProcessError:
            output = b''
        outputs.append(output)
    py_output, mg_output = outputs

    if py_output != mg_output:
        fails += 1
        print('PYTHON TEST %s FAILED!' % test)
        with tempfile.NamedTemporaryFile() as f1:
            f1.write(py_output)
            f1.flush()
            with tempfile.NamedTemporaryFile() as f2:
                f2.write(mg_output)
                f2.flush()
                subprocess.call(['diff', '-a', f1.name, f2.name])
    else:
        passes += 1

for test in raw_tests:
    if test_set and test not in test_set:
        continue

    # Run python script, just check the output code
    try:
        path = '%s.py' % (test)
        subprocess.check_call([python, path], cwd=raw_test_dir)
        passes += 1
    except Exception:
        fails += 1
        print('TEST %s FAILED!' % test)

end = time.time()
print('%s/%s tests passed. Time: %.2fs' % (passes, passes + fails, end - start))
sys.exit(fails != 0)
