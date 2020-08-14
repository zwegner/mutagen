#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile
import time

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.environ['PYTHONPATH'] = root_dir

mutagen = '%s/bootstrap/mutagen.py' % root_dir
python = 'python3'

passes = fails = 0

exclude_set, test_set = [[arg.replace('-', '') for arg in sys.argv[1:]
    if arg.startswith('-') ^ exclude] for exclude in range(2)]

# Mutagen tests: these should just not throw any errors
mg_tests = ['builtins', 'core', 'effects']
mg_test_dir = 'tests/mg'

# Python tests: python/mutagen should produce the same output in each case.
py_tests = ['core', 'regex', 'lex']
py_test_dir = 'tests/py'

# Raw tests: these are custom Python scripts that run arbitrary code, and
# should exit cleanly
raw_tests = ['asm', 'regalloc']
raw_test_dir = 'tests/raw'

start = time.time()

def active_tests(tests):
    return [test for test in tests
            if (not test_set or any(pattern in test for pattern in test_set)) and
            not any(pattern in test for pattern in exclude_set)]

# Run mutagen tests
for test in active_tests(mg_tests):
    try:
        path = '%s.mg' % (test)
        subprocess.check_output([mutagen, path], cwd=mg_test_dir)
        passes += 1
    except Exception:
        fails += 1
        print('MUTAGEN TEST %s FAILED!' % test)

for test in active_tests(py_tests):
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
        if py_output and mg_output:
            with tempfile.NamedTemporaryFile() as f1:
                f1.write(py_output)
                f1.flush()
                with tempfile.NamedTemporaryFile() as f2:
                    f2.write(mg_output)
                    f2.flush()
                    subprocess.call(['diff', '-a', f1.name, f2.name])
    else:
        passes += 1

for test in active_tests(raw_tests):
    # Run python script, just check the output code
    try:
        path = '%s.py' % (test)
        subprocess.check_call([python, path], cwd=raw_test_dir)
        passes += 1
    except Exception:
        fails += 1
        print('RAW TEST %s FAILED!' % test)

end = time.time()
print('%s/%s tests passed. Time: %.2fs' % (passes, passes + fails, end - start))
sys.exit(fails != 0)
