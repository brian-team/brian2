'''
Brian 2.0
'''
# Check basic dependencies
import sys
missing = []
try:
    import numpy
except ImportError as ex:
    sys.stderr.write('Importing numpy failed: %s\n' % ex)
    missing.append('numpy')
try:
    import scipy
except ImportError as ex:
    sys.stderr.write('Importing scipy failed: %s\n' % ex)
    missing.append('scipy')
try:
    import sympy
except ImportError as ex:
    sys.stderr.write('Importing sympy failed: %s\n' % ex)
    missing.append('sympy')
try:
    import pyparsing
except ImportError as ex:
    sys.stderr.write('Importing pyparsing failed: %s\n' % ex)
    missing.append('pyparsing')

if len(missing):
    raise ImportError('Some required dependencies are missing:\n' + ', '.join(missing))

try:
    from pylab import *
except ImportError:
    from scipy import *

# Make sure that Brian's unit-aware functions are used, even when directly
# using names prefixed with numpy or np
import brian2.numpy_ as numpy
import brian2.numpy_ as np

# delete some annoying names from the namespace
if 'x' in globals():
    del x
if 'f' in globals():
    del f
if 'rate' in globals():
    del rate

__docformat__ = "restructuredtext en"

__version__ = '2.0a6'
__release_date__ = '2013-12-18'

from brian2.only import *
