from brian2 import *

brian_prefs.codegen.target = 'weave'

import numpy as np

G = NeuronGroup(10, 'x:1')
G.x = np.arange(10)
print G.x
SG = G[4:]
print SG.i[:]
S = Synapses(SG, G, 'w:1', connect='i==j')
print S.i[:]