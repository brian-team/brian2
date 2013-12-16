'''
Brian 2.0
'''
__docformat__ = "restructuredtext en"

__version__ = '2.0a5'
__release_date__ = '2013-10-28'

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

# delete some annoying names from the namespace
if 'x' in globals():
    del x
if 'f' in globals():
    del f
if 'rate' in globals():
    del rate

from brian2.only import *
