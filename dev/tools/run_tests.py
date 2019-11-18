'''
Run all the non-standalone tests using nose. Exits with error code 1 if a test failed.
'''
import sys

import brian2

if __name__ == '__main__':
    if not brian2.test():  # If the test fails, exit with a non-zero error code
        sys.exit(1)
