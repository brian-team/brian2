try:
    import scipy as sp
except ImportError:
    sp = None

from brian2 import *

from brian2.units.fundamentalunits import DimensionMismatchError, Quantity

def print_eval(expr):
    print expr, '=', eval(expr)

# dimensions with scalars should work as before
print '\tScalars with units' 
print_eval('3 * mV')
print_eval('mV * 3')
print_eval('3 * mV/mV')
print_eval('(23 * metre) ** 2')
print_eval('np.sqrt((23 * metre) ** 2)')
print_eval('np.sin(2 * np.pi * 100 * Hz * 1 * second)')
# This did actually not work before
print_eval('np.float32(23) * (1 * metre)')

# This should raise an exceptions
try:
    np.sin(3 * second)
except DimensionMismatchError as ex:
    print 'Expected error raised: ', ex

print '\n\tRemoving units'
# Using float, array or asarray removes units
print_eval('float(23 * cm)')
print_eval('np.float32(23 * cm)')
print_eval('np.array(23 * cm)')
print_eval('np.asarray(23 * cm)')

# This is new, arrays with dimensions
print '\n\tArrays with units' 
print_eval('np.array([1, 2, 3]) * mV')
print_eval('np.array([1, 2, 3]) * mV/mV')
print_eval('(np.array([1, 2, 3]) * metre) ** 2')
print_eval('np.sqrt((np.array([1, 2, 3]) * metre) ** 2)')
print_eval('np.sin(2 * np.pi * 100 * Hz * np.array([0, 2.5, 5]) * ms)')
print_eval('np.array([1, 2, 3], dtype=np.int32) * metre')

print '\n\tremoving units'
print_eval('np.asarray(np.array([1, 2, 3]) * metre)')

# Calculating with unit arrays
print '\n\tCalculations with unit arrays'
print_eval('np.array([10, 20, 30]) * cm + 1 * metre')
print_eval('np.array([10, 20, 30]) * cm + np.array([1, 2, 3]) * metre')
print_eval('np.array([10, 20, 30]) * cm * (1 / second)')
print_eval('np.array([10, 20, 30]) * cm * (np.array([1, 2, 3]) / second)')
print_eval('np.array([10, 20, 30]) * cm / (np.array([1, 2, 3]) * second)')
print_eval('np.array([10, 20, 30]) * cm * 5')

# Multidimensional arrays
print_eval('np.ones((3, 4)) * mV')
print_eval('(np.ones((3, 4)) * mV).flatten()')

# Unit mismatches
print '\nUnit mismatches'
try:
    np.array([10, 20, 30]) * cm + 3 * second
except DimensionMismatchError as ex:
    print 'Expected error raised: ', ex

try:
    np.array([10, 20, 30]) * cm + np.array([1, 2, 3]) * second
except DimensionMismatchError as ex:
    print 'Expected error raised: ', ex

try:
    np.sin(np.array([10, 20, 30]) * cm)
except DimensionMismatchError as ex:
    print 'Expected error raised: ', ex

# More complicated functions
print '\n\tnumpy functions'
print_eval('(np.array([10, 20, 30]) * cm).mean()')
print_eval('np.mean(np.array([10, 20, 30]) * cm)')
print_eval('(np.array([10, 20, 30]) * cm).var()')
print_eval('np.var(np.array([10, 20, 30]) * cm)')
print_eval('np.max(np.array([10, 20, 30]) * cm)')
print_eval('np.sum(np.array([10, 20, 30]) * cm)')
print_eval('np.clip(np.array([10, 20, 30]) * cm, 20 * cm, np.Inf * cm)')
# print '\nShould this have units of cm**3?'
# print_eval('np.prod(np.array([10, 20, 30]) * cm)')

print '\n\tComparisons'
print_eval('np.array([10, 20, 30]) * cm == np.array([0.2, 0.2, 0.2]) * metre')
print_eval('np.array([10, 20, 30]) * cm < np.array([0.2, 0.2, 0.2]) * metre')
try:
    np.array([10, 20, 30]) * cm < np.array([0.2, 0.2, 0.2]) * second
except DimensionMismatchError as ex:
    print 'Expected error raised: ', ex

# Slicing and indexing
print '\n\tSlicing and indexing'
print_eval('(np.array([10, 20, 30]) * cm)[:]')
print_eval('(np.array([10, 20, 30]) * cm)[0:2]')
print_eval('(np.array([10, 20, 30]) * cm)[0]')
print_eval('(np.array([10, 20, 30]) * cm)[np.array([0, 2])]')
print_eval('(np.array([10, 20, 30]) * cm)[np.array([True, False, True])]')

print '\n\tAssigning values'
ar = np.array([10, 20, 30]) * cm
print 'Before assignment: ', ar
ar[0] = 20 * cm
print 'After assignment: ', ar

ar[0:2] = np.array([50, 50]) * cm
print 'After assignment: ', ar

try:
    ar[1] = 2 * second
except DimensionMismatchError as ex:
    print 'Expected error raised: ', ex

try:
    ar[0:2] = 2 * second
    print ar
except DimensionMismatchError as ex:
    print 'Expected error raised: ', ex

print '\n\tConversions to and from list'
print_eval('Quantity([1 * mV, 1 * volt])')
print_eval('(np.array([1, 2, 3]) * mV).tolist()')

print '\n\tExamples of functions that are not unit-aware'
print_eval('np.histogram(np.array([1, 2, 3]) * mV)')
print_eval('np.correlate(np.array([1, 2, 3]) * mV, np.array([1, 2, 3]) * mV)')
print_eval('np.trapz(np.array([1, 2, 3]) * mV, np.array([1, 2, 3]) * second)')
print_eval('np.arange(0 * mV, 10 * mV, 1 * mV)')
if sp is not None:
    print_eval('sp.sqrt(np.array([1.5, 3]) * mV)')
    print_eval('sp.interp(np.array([1.5]) * mV, np.array([1, 2, 3]) * mV, np.array([1, 2, 3]) * second)')
    print_eval('sp.fft(np.array([1.5, 3]) * mV)')
    print_eval('sp.average(np.array([1.5, 3]) * mV, weights=[1, 2])')
