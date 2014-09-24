from brian2 import *

brian_prefs.codegen.target = 'weave'
#brian_prefs.codegen.target = 'cython'

G = NeuronGroup(10, 'v : 1')
#S = Synapses(G, G, 'w:1', connect='i==j')
#S.w = 'i'
#print S.w

#G.v = 'i'
#print G.v

v = G.v
v[5:] = 'i'
print G.v
