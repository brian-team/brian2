from brian2 import *

brian_prefs['codegen.target'] = 'weave'
G = NeuronGroup(1, 'v:1', threshold='v>1', refractory=1*ms)
run(0*ms)
print G.thresholder.codeobj.code.main