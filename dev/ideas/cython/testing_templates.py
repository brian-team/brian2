from brian2 import *

#brian_prefs.codegen.target = 'weave'
brian_prefs.codegen.target = 'cython'

G = SpikeGeneratorGroup(2, [0, 1, 0, 1], [0*ms, 1*ms, 2*ms, 3*ms])
M = SpikeMonitor(G)
run(4*ms)
print M.i, M.t
