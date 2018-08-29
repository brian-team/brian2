import sys

import brian2
import numpy as np

# Run tests for float32 and float64
success = [brian2.test(long_tests=True, test_standalone='cpp_standalone', float_dtype=np.float32),
           brian2.test(long_tests=True, test_standalone='cpp_standalone', float_dtype=np.float64)]

result = ['Tests for {} dtype: {}'.format(dtype,
                                         'passed' if status else 'FAILED')
          for status, dtype in zip(success, ['float32', 'float64'])]
print('\t--\t'.join(result))

if all(success):
    print('OK: All tests passed successfully')
else:
    print('FAILED: Not all tests passed successfully (see above)')
    sys.exit(1)

