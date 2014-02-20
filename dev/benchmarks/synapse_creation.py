'''
How long does synapse creation take?
'''
import time
import cPickle

import numpy as np
import joblib

from brian2 import *

repetitions = 3

memory = joblib.Memory(cachedir='.', verbose=0)

@memory.cache
def test_connectivity2(N, i, j, n, p, codeobj_class):
    G = NeuronGroup(N, '')
    # Do it once without measuring the time to ignore the compilation time for
    # C code
    S = Synapses(G, G, '', codeobj_class=codeobj_class)
    S.connect(i, j, p, n)
    connections = len(S)
    del S
    times = []
    for _ in xrange(repetitions):
        S = Synapses(G, G, '', codeobj_class=codeobj_class)
        start = time.time()
        S.connect(i, j, p, n)
        times.append(time.time() - start)
        del S

    return float(np.median(times)), connections

conditions = [('Full', 'True'),
              ('Full (no-self)', 'i != j'),
              ('One-to-one', 'i == j'),
              ('Simple neighbourhood', 'abs(i-j) < 5'),
              ('Gauss neighbourhood', 'exp(-(i - j)**2/5) > 0.005'),
              ('Random (50%)', (True, None, 1, 0.5)),
              ('Random (10%)', (True, None, 1, 0.1)),
              ('Random (1%)', (True, None, 1, 0.01)),
              ('Random no-self (50%)', ('(i != j)', None, 1, 0.5)),
              ('Random no-self (10%)', ('(i != j)', None, 1, 0.1)),
              ('Random no-self (1%)', ('(i != j)', None, 1, 0.01))]
targets = [NumpyCodeObject, WeaveCodeObject]
results = {}
max_connections = 2500000
for target in targets:
    lang_name = target.class_name
    for pattern, condition in conditions:
        N = 1
        connections = took = 0
        while connections < max_connections and took < 20:
            print lang_name, pattern
            if isinstance(condition, basestring):
                took, connections = test_connectivity2(N, condition, None, 1, 1.,
                                                      codeobj_class=target)
            else:
                took, connections = test_connectivity2(N, *condition,
                                                      codeobj_class=target)
            print N, '%.4fs (for %d connections)' % (took, connections)
            results[(lang_name, connections, pattern)] = took
            N *= 2

with open('synapse_creation_times_brian2.pickle', 'w') as f:
    cPickle.dump(results, f)
