#!/usr/bin/env python3
import os
import subprocess
import sys
import time

bench_dir = sys.path[0]
root_dir = os.path.dirname(bench_dir)
mutagen = '%s/bootstrap/parse.py' % root_dir
python = 'python3'

benchmarks = ['pidigits']

fails = 0

fmt = '| %-10s | %-9s | %-9s | %-9s |'
sep = '-' * 50
nfmt = '%.2fs'
print(sep)
print(fmt % ('name', 'mutagen', 'python', 'mg/py'))
print(sep)

for bm in benchmarks:
    path = '%s.mg' % bm

    speed = {}
    output = {}
    for language in [mutagen, python]:
        start = time.time()
        output[language] = subprocess.check_output([language, path], cwd=bench_dir)
        end = time.time()
        speed[language] = (end - start)

    if output[python] != output[mutagen]:
        fails += 1
        print('BENCHMARK %s OUTPUT MISMATCH!' % bm)
    else:
        print(fmt % (bm, nfmt % speed[mutagen], nfmt % speed[python],
            '%.2fx' % (speed[mutagen] / speed[python])))

print(sep)

sys.exit(fails != 0)
