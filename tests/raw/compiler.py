#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile

this_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath('%s/../..' % this_dir)
sys.path.append(root_dir)

import cffi

from bootstrap import compiler
from bootstrap import syntax
from bootstrap.sprdpl import parse as libparse

# Define a @test_fn decorator, and a _test_expected_value(value, fn) helper
# intrinsic (needed so test_fn can act as a partial function: @test_fn(4)(fn)
# is equivalent to _test_expected_value(4, fn))

TEST_FNS = []
@compiler.mg_intrinsic([syntax.Integer, syntax.Function])
def mgi__test_expected_value(node, expected, fn):
    # Set the 'expected' attribute on the function, and also keep it in a separate
    # list so we can double-check later that all the test functions got compiled
    fn.attributes['expected'] = expected.value
    TEST_FNS.append(fn)
    return fn

@compiler.mg_intrinsic([syntax.Integer])
def mgi_test_fn(node, expected):
    node = syntax.PartialFunction(mgi__test_expected_value, [expected])
    compiler.add_node_usages(node)
    return node

def run(path):
    # Compile the input file
    fns = compiler.compile_file(path)

    # Export all the @test_fn()-decorated functions, and the @export'ed ones too
    export_fns = [fn for fn in fns if 'expected' in fn.attributes or
            fn.attributes.get('export')]
    test_fns = [fn for fn in fns if 'expected' in fn.attributes]
    # Verify that we got the same list of test functions through both accounting methods
    assert test_fns == TEST_FNS, (test_fns, TEST_FNS)

    with tempfile.TemporaryDirectory() as tmp_dir:
        obj_file = '%s/test_out.o' % tmp_dir
        so_file = '%s/test_lib.so' % tmp_dir

        # Compile functions
        compiler.export_functions(obj_file, export_fns)

        # Convert to Mach-O object files on macOS, using the objconv library
        # from Agner Fog (have to build/install this manually...)
        if sys.platform == 'darwin':
            obj_file_2 = '%s/test_out2.o' % tmp_dir
            subprocess.check_call(['./objconv/source/objconv',
                    '-v0', '-fmacho64', obj_file, obj_file_2])
            obj_file = obj_file_2

        # Convert .o to .so
        subprocess.check_call(['cc', obj_file, '-shared', '-o', so_file])

        # Define prototypes for each test function
        prototypes = '\n'.join('uint64_t %s(void);' % fn.name for fn in test_fns)
        ffi = cffi.FFI()
        ffi.cdef(prototypes)

        # Import the library
        lib = ffi.dlopen(so_file)

    # Run each test function and check the result
    any_failed = False
    for fn in test_fns:
        ffi_fn = getattr(lib, fn.name)
        expected = fn.attributes.get('expected')
        value = ffi_fn()
        if value != expected:
            print('FAIL: test %s returned %s, expected %s' % (fn.name, value, expected))
            any_failed = True
    return any_failed

def main():
    path = '%s/compiler/basic.mg' % this_dir
    try:
        sys.exit(run(path))
    except (libparse.ParseError, syntax.ProgramError) as e:
        e.print()
        sys.exit(1)

if __name__ == '__main__':
    main()
