from brian2 import *

#brian_prefs.codegen.target = 'weave'
brian_prefs.codegen.target = 'cython'

G = NeuronGroup(10, 'v : 1')

G.v['i>5'] = 'i'
print G.v
