#!/usr/bin/env python
'''
The most simple case to shwo how to use standalone mode.
'''


import numpy as np
import brian2 as b2
import pylab as plt

b2.start_scope()
b2.set_device('cpp_standalone', build_on_run=False)
# b2.prefs.codegen.target = 'numpy'  # use the Python fallback

tau = 10*b2.ms
eqs = '''
dv/dt = (1-v)/tau : 1
'''
G = b2.NeuronGroup(1, eqs, method='exact')
b2.run(100*b2.ms)
b2.device.build(directory='output', compile=True, run=True, debug=False)
print('After v = %s' % G.v[0])