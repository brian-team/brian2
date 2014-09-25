from brian2 import *

#brian_prefs.codegen.target = 'weave'
brian_prefs.codegen.target = 'cython'

G = NeuronGroup(1000, 'dv/dt = 1/(t+1*ms) : 1', threshold='v>1', reset='v=0')
G.v = 'rand()'
R = PopulationRateMonitor(G)
run(1000*ms)
plot(R.t, R.rate)
show()
