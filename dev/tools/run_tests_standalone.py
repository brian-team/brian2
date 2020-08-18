'''
Run all the standalone tests using nose. Exits with error code 1 if a test failed.
'''
import sys

import brian2

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'no-parallel':
        if not brian2.test([], test_codegen_independent=False,
                           test_standalone='cpp_standalone',
                           test_in_parallel=[]):  # If the test fails, exit with a non-zero error code
            sys.exit(1)
    else:
        if not brian2.test([], test_codegen_independent=False,
                           test_standalone='cpp_standalone'):  # If the test fails, exit with a non-zero error code
            sys.exit(1)
