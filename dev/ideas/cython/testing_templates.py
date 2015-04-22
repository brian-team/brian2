from brian2 import *

#brian_prefs.codegen.target = 'weave'
brian_prefs.codegen.target = 'cython'

G = NeuronGroup(2, 'v:boolean')
M = StateMonitor(G, 'v', record=True)
run(10*ms)
G.v = True
run(10*ms)
print M.v[0]
