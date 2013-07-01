'''
How long does synapse creation take?
'''
import time

import numpy as np
import joblib

from brian2 import *

repetitions = 5

memory = joblib.Memory(cachedir='.', verbose=0)

@memory.cache
def test_connectivity(N, i, j, n, p, language):
    G = NeuronGroup(N, '')
    # Do it once without measuring the time to ignore the compilation time for
    # C code
    S = Synapses(G, G, '', language=language)
    S.connect(i, j, p, n)
    del S
    times = []
    for _ in xrange(repetitions):
        S = Synapses(G, G, '', language=language)
        start = time.time()
        S.connect(i, j, p, n)
        times.append(time.time() - start)
        del S

    return np.median(times)

# Full connectivity
results = {}
for language in [PythonLanguage(), CPPLanguage()]:
    lang_name = language.__class__.__name__
    for pattern, condition in [('Full', True),
                               ('Full (no-self)', 'i != j'),
                               ('One-to-one', 'i == j'),
                               ('Neighbourhood', 'abs(i-j) < 5')]:
        for N in [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]:
            print lang_name, pattern
            took = test_connectivity(N, condition, None, 1, 1., language=language)
            print N, '%.4fs' % took
            results[(lang_name, N, pattern)] = took

