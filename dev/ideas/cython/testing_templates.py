from brian2 import *

brian_prefs.codegen.target = 'cython'

G = NeuronGroup(10, 'v:1')

#G.v = 'i'
#
print G.indices['i>=0']
