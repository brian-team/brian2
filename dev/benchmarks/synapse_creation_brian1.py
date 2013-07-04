'''
How long does synapse creation take? Brian1 for comparison
'''
import time
import cPickle

import numpy as np
import joblib

from brian import *

repetitions = 5

memory = joblib.Memory(cachedir='.', verbose=0)

@memory.cache
def test_connectivity(N, i, j, n, p):
    G = NeuronGroup(N, '')
    # Do it once without measuring the time to ignore the compilation time for
    # C code -- not really necessary for Brian1 but leave it in for perfect
    # comparability
    S = Synapses(G, G, '')
    S[:, :] = i
    connections = len(S)
    del S
    times = []
    for _ in xrange(repetitions):
        S = Synapses(G, G, '')
        start = time.time()
        S[:, :] = i
        times.append(time.time() - start)
        del S

    return np.median(times), connections

conditions = [('Full', True),
              ('Full (no-self)', 'i != j'),
              ('One-to-one', 'i == j'),
              ('Simple neighbourhood', 'abs(i-j) < 5'),
              ('Gauss neighbourhood', 'exp(-(i - j)**2/5) > 0.005'),
              ('Random (50%)', 0.5),
              ('Random (10%)', 0.1),
              ('Random (1%)', 0.01),
              ('Random no-self (50%)', '(i != j) * 0.5'),
              ('Random no-self (10%)', '(i != j) * 0.1'),
              ('Random no-self (1%)', '(i != j) * 0.01')]
results = {}
lang_name = 'Brian 1'
max_connections = 10000000
for pattern, condition in conditions:
    N = 1
    connections = took = 0
    while connections < max_connections and took < 60.:
        print lang_name, pattern
        took, connections = test_connectivity(N, condition, None, 1, 1.)
        print N, '%.4fs (for %d connections)' % (took, connections)
        results[(lang_name, connections, pattern)] = took
        N *= 2

with open('synapse_creation_times_brian1.pickle', 'w') as f:
    cPickle.dump(results, f)
