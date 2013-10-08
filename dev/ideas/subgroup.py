from brian2 import *

brian_prefs.codegen.target = 'weave'
brian_prefs['codegen.runtime.weave.extra_compile_args'] = ['-g', '-O0']
import numpy as np

G = NeuronGroup(20, 'x:1')
G.x = np.arange(20)
print G.x
SG = G[4:15]
print SG.i[:]
S = Synapses(SG, G, 'w:1', connect='i==j')
print S.i[:]
print S.j[:]