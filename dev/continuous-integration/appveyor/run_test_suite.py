'''
Simple script to run the test suite from appveyor.yml (easier than putting all
the logic into Windows batch statements)
'''

import os
import sys

import brian2

split_run = os.environ.get('SPLIT_RUN', None)
standalone = os.environ.get('STANDALONE', 'FALSE').lower() == 'true'

if split_run == '1':
    targets = ['numpy', 'weave']
    independent = True
elif split_run == '2':
    targets = ['cython']
    independent = False
else:
    targets = None
    independent = True

if standalone:
    result = brian2.test([],
                         test_codegen_independent=False,
                         test_standalone='cpp_standalone',
                         test_in_parallel=[])
else:
    result = brian2.test(targets,
                         test_codegen_independent=independent,
                         test_standalone=None,
                         test_in_parallel=[])

if not result:
    sys.exit(1)
