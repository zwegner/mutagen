#!/usr/bin/env python3
import os
import subprocess

groups = ['bootstrap/sprdpl', 'bootstrap', 'stdlib', 'tests', 'compiler', 'benchmarks', '']
group_dict = {k: [] for k in groups}

def get_files_recursively(root):
    env = {'GIT_DIR': '%s/.git' % root}
    files = subprocess.check_output(['git', 'ls-files'], env=env).decode('utf8').splitlines()

    for f in files:
        if os.path.isdir(f):
            get_files_recursively(f)
        else:
            for g in groups:
                path = os.path.normpath('%s/%s' % (root, f))
                if path.startswith(g):
                    group_dict[g].append(path)
                    break

get_files_recursively('.')

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

