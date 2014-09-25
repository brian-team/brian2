from brian2 import *

#brian_prefs.codegen.target = 'weave'
brian_prefs.codegen.target = 'cython'

G = NeuronGroup(10, 'v : 1')
S = SpikeMonitor(G)
run(1*ms)
