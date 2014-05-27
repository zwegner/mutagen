#!/usr/bin/env python3
import subprocess

files = subprocess.check_output(['git', 'ls-files']).decode('utf8').splitlines()

groups = ['bootstrap', 'stdlib', 'tests', 'compiler', 'benchmarks', '']
group_dict = {k: [] for k in groups}

for f in files:
    for g in groups:
        if f.startswith(g):
            group_dict[g].append(f)
            break

total = [0, 0]
for group, files in sorted(group_dict.items()):
    print('%s: ' % group)
    group_total = [0, 0]
    for file in files:
        with open(file) as f:
            data = f.read()
        bytes = len(data)
        lines = len(data.splitlines())
        print('    %6sl %6sb %s' % (lines, bytes, file))
        group_total[0] += lines
        group_total[1] += bytes
    print('    %6sl %6sb -> total' % (group_total[0], group_total[1]))
    total[0] += group_total[0]
    total[1] += group_total[1]
print('%6sl %6sb -> total' % (total[0], total[1]))

