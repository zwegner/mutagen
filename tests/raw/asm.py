#!/usr/bin/env python3
import os
import re
import subprocess
import sys
import tempfile

root_dir = os.path.dirname(os.path.dirname(sys.path[0]))

# Create a temporary directory to take care of any files we create
with tempfile.TemporaryDirectory() as tmp_dir:
    os.chdir(tmp_dir)

    raw_output = subprocess.check_output(['%s/bootstrap/parse.py' % root_dir,
        '%s/compiler/asm.mg' % root_dir]).decode('ascii')

    dump_output = subprocess.check_output(['gobjdump', '--no-show-raw-insn',
        '-Mintel', '-D', 'elfout.o']).decode('ascii')

# Munge the output so the output formats mostly match
lines = []
for line in dump_output.splitlines():
    for [sub, rep] in [
            [r'#.*', ''], # comments (specify rip+disp locations)
            [r'\s+', ' '], # multiple whitespace
            [r'\s+$', ''], # trailing whitespace
            ['movabs', 'mov'], # instruction munging
            ['0x[0-9a-f]+', lambda m: str(int(m.group(0), 16))], # hex->dec
            [r'\+0]', ']']]: # get rid of 0 disp from RBP/R13 base
        line = re.sub(sub, rep, line)
    # Check for a label line
    m = re.search(r'^[0-9a-f]+ <([_0-9A-Za-z]+)>:$', line)
    if m:
        lines.append(m.group(1) + ':')
    else:
        # Check for instructions
        m = re.search(r'^ *[0-9a-f]+:\s*([0-9A-Za-z,+*\[\] ]+)$', line)
        if m:
            lines.append('    ' + m.group(1))
lines.append('')
dump_output = '\n'.join(lines)

# Exit codes are reversed from boolean comparisons (0 is successful)
sys.exit(raw_output != dump_output)
