'''
Script to run the test suite during automatic testing (easier than putting all
the logic into Windows batch/bash statements)
'''
# Importing multiprocessing here seems to fix hangs in the test suite on OS X
# see https://github.com/scipy/scipy/issues/11835
import multiprocessing
import os
import sys

import numpy as np

import brian2

if __name__ == '__main__':
    split_run = os.environ.get('SPLIT_RUN', None)
    standalone = os.environ.get('STANDALONE', 'no').lower() in ['yes', 'true']
    python_version = os.environ.get('PYTHON_VERSION', os.environ.get('PYTHON'))
    # If TRAVIS_OS_NAME is not defined, we are testing on appveyor
    operating_system = os.environ.get('AGENT_OS', 'unknown').lower()
    cross_compiled = os.environ.get('CROSS_COMPILED', 'FALSE').lower() in ['yes', 'true']
    report_coverage = os.environ.get('REPORT_COVERAGE', 'no').lower() in ['yes', 'true']
    dtype_32_bit = os.environ.get('FLOAT_DTYPE_32', 'no').lower() in ['yes', 'true']
    sphinx_dir = os.environ.get('SPHINX_DIR')
    src_dir = os.environ.get('SRCDIR')

    if split_run == '1':
        targets = ['numpy']
        independent = True
    elif split_run == '2':
        targets = ['cython']
        independent = False
    else:
        targets = None
        independent = True

    if operating_system == 'windows' or standalone:
        in_parallel = []
    else:
        in_parallel = ['codegen_independent', 'numpy', 'cpp_standalone']

    if operating_system in ['linux', 'windows_nt']:
        openmp = True
    else:
        openmp = False

    reset_preferences = not cross_compiled
    if dtype_32_bit:
        float_dtype = np.float32
    else:
        float_dtype = None

    if standalone:
        result = brian2.test([],
                             test_codegen_independent=False,
                             test_standalone='cpp_standalone',
                             test_openmp=openmp,
                             test_in_parallel=in_parallel,
                             reset_preferences=reset_preferences,
                             float_dtype=float_dtype,
                             test_GSL=True,
                             sphinx_dir=sphinx_dir)
    else:
        result = brian2.test(targets,
                             test_codegen_independent=independent,
                             test_standalone=None,
                             test_in_parallel=in_parallel,
                             reset_preferences=reset_preferences,
                             float_dtype=float_dtype,
                             test_GSL=True,
                             sphinx_dir=sphinx_dir)

    if not result:
        sys.exit(1)
