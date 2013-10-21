'''
Some simple benchmarks to check how much the performance is impaired when using units (units are not used in the main
simulation loop but also analysing/plotting should not be noticeably slow for the user)
'''
import time

import numpy as np

from brian2 import *

sizes = [10, 100, 1000, 10000, 100000]

access = lambda idx, ar1, ar2: ar1[idx]
multiply = lambda idx, ar1, ar2: 42 * ar1
addition = lambda idx, ar1, ar2: ar1 + ar2
slicing = lambda idx, ar1, ar2: ar1[idx:]

for func_name, func in [('access', access),
                        ('multiply', multiply),
                        ('add', addition),
                        ('slicing', slicing)]:
    times_no_unit = []
    times_unit = []
    for size in sizes:
        no_unit1 = np.random.randn(size)
        no_unit2 = np.random.randn(size)
        with_unit1 = no_unit1 * mV
        with_unit2 = no_unit2 * mV
        start = time.time()
        for x in xrange(size):
           func(x, no_unit1, no_unit2)
        times_no_unit.append(time.time() - start)
        start = time.time()
        for x in xrange(size):
            func(x, with_unit1, with_unit2)
        times_unit.append(time.time() - start)
    print ''
    print func_name,':'
    print 'No unit ', times_no_unit
    print '   unit ', times_unit
    print 'relative', np.array(times_unit) / np.array(times_no_unit)

# Dimensionless Quantities
print '*' * 60
print 'Dimensionless quantities'

for func_name, func in [('access', access),
                        ('multiply', multiply),
                        ('add', addition),
                        ('slicing', slicing)]:
    times_no_unit = []
    times_unit = []
    for size in sizes:
        no_unit1 = np.random.randn(size)
        no_unit2 = np.random.randn(size)
        with_unit1 = no_unit1 * mV/mV
        with_unit2 = no_unit2 * mV/mV
        start = time.time()
        for x in xrange(size):
           func(x, no_unit1, no_unit2)
        times_no_unit.append(time.time() - start)
        start = time.time()
        for x in xrange(size):
            func(x, with_unit1, with_unit2)
        times_unit.append(time.time() - start)
    print ''
    print func_name,':'
    print 'No unit ', times_no_unit
    print '   unit ', times_unit
    print 'relative', np.array(times_unit) / np.array(times_no_unit)