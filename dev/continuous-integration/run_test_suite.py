'''
Script to run the test suite during automatic testing (easier than putting all
the logic into Windows batch/bash statements)
'''

import os
import sys

import brian2

split_run = os.environ.get('SPLIT_RUN', None)
standalone = os.environ.get('STANDALONE', 'no').lower() in ['yes', 'true']
# If TRAVIS_OS_NAME is not defined, we are testing on appveyor
operating_system = os.environ.get('TRAVIS_OS_NAME', 'windows').lower()
cross_compiled = os.environ.get('CROSS_COMPILED', 'FALSE').lower() in ['yes', 'true']
report_coverage = os.environ.get('REPORT_COVERAGE', 'no').lower() in ['yes', 'true']

if split_run == '1':
    targets = ['numpy', 'weave']
    independent = True
elif split_run == '2':
    targets = ['cython']
    independent = False
else:
    targets = None
    independent = True

if operating_system == 'windows' or report_coverage:
    in_parallel = []
else:
    in_parallel = ['codegen_independent', 'numpy', 'cython', 'cpp_standalone']

if operating_system in ['linux', 'windows']:
    openmp = True
else:
    openmp = False

reset_preferences = not cross_compiled
if standalone:
    result = brian2.test([],
                         test_codegen_independent=False,
                         test_standalone='cpp_standalone',
                         test_openmp=openmp,
                         test_in_parallel=in_parallel,
                         reset_preferences=reset_preferences)
else:
    result = brian2.test(targets,
                         test_codegen_independent=independent,
                         test_standalone=None,
                         test_in_parallel=in_parallel,
                         reset_preferences=reset_preferences)

if not result:
    sys.exit(1)
