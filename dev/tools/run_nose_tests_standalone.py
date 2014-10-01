'''
Run all the standalone tests using nose. Exits with error code 1 if a test failed.
'''
import sys

import brian2

if not brian2.test([], test_standalone=True):  # If the test fails, exit with a non-zero error code
    sys.exit(1)
