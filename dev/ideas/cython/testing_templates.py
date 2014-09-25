from brian2 import *

#brian_prefs.codegen.target = 'weave'
brian_prefs.codegen.target = 'cython'

G = NeuronGroup(10, 'dv/dt = 1/second : 1')
print G.indices['v>=0']
#M = StateMonitor(G, 'v', record=True)
#run(10*ms)
#print M.t
#plot(M.t/ms, M.v[0])
#show()
