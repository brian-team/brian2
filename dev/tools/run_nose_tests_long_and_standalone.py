import sys

import brian2

if not brian2.test(long_tests=True, test_standalone=True):
    sys.exit(1)
