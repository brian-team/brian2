from brian2 import *

#brian_prefs.codegen.target = 'weave'
brian_prefs.codegen.target = 'cython'

G = NeuronGroup(10, 'v = i : 1')

#G.v = 'i'
#
print G.v#['i>=0']
