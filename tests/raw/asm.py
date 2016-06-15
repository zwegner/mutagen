#!/usr/bin/env python3
import os
import re
import subprocess
import sys
import tempfile

root_dir = os.path.dirname(os.path.dirname(sys.path[0]))

for exec_path in [['python3', '%s/tests/raw/test_asm.py' % root_dir],
        ['%s/bootstrap/parse.py' % root_dir, '%s/tests/raw/test_asm.mg' % root_dir]]:
    # Create a temporary directory to take care of any files we create
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        raw_output = subprocess.check_output(exec_path).decode('ascii')

        dump_output = subprocess.check_output(['gobjdump', '--no-show-raw-insn',
            '-Mintel', '-D', 'elfout.o']).decode('ascii')

    # Munge the output so the output formats mostly match
    lines = []
    for line in dump_output.splitlines():
        for [sub, rep] in [
                ['#.*', ''], # comments (specify rip+disp locations)
                ['\\s+', ' '], # multiple whitespace
                ['\\s+$', ''], # trailing whitespace
                ['movabs', 'mov'], # instruction munging
                [' [0-9a-f]+ (<[_0-9A-Za-z]+>)', ' \\1'], # remove raw address of labels
                ['0x[0-9a-f]+', lambda m: str(int(m.group(0), 16))], # hex->dec
                ['\\+0]', ']']]: # get rid of 0 disp from RBP/R13 base
            line = re.sub(sub, rep, line)
        # Check for a label line
        m = re.search(r'^[0-9a-f]+ <([_0-9A-Za-z]+)>:$', line)
        if m:
            lines.append(m.group(1) + ':')
        else:
            # Check for instructions
            m = re.search(r'^ *[0-9a-f]+:\s*([_0-9A-Za-z,+*\[\]<> ]+)$', line)
            if m:
                lines.append('    ' + m.group(1))
    lines.append('')
    dump_output = '\n'.join(lines)

    if raw_output != dump_output:
        with tempfile.NamedTemporaryFile() as f1:
            f1.write(raw_output.encode('ascii'))
            f1.flush()
            with tempfile.NamedTemporaryFile() as f2:
                f2.write(dump_output.encode('ascii'))
                f2.flush()
                subprocess.call(['diff', f1.name, f2.name])
        sys.exit(1)

sys.exit(0)
