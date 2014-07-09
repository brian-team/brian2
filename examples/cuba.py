from brian2 import *
import time

# BrianLogger.log_level_debug()
#brian_prefs.codegen.target = 'weave'

taum = 20 * ms
taue = 5 * ms
taui = 10 * ms
Vt = -50 * mV
Vr = -60 * mV
El = -49 * mV

eqs = '''
dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : volt (unless refractory)
dgi/dt = -gi/taui : volt (unless refractory)
'''

P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v=Vr', refractory=5*ms)
P.v = Vr
P.ge = 0 * mV
P.gi = 0 * mV

we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight
Ce = Synapses(P, P, 'w:1', pre='ge += we')
Ci = Synapses(P, P, 'w:1', pre='gi += wi')
Ce.connect('i<3200', p=0.02)
Ci.connect('i>=3200', p=0.02)
P.v = Vr + rand(len(P)) * (Vt - Vr)

start_time = time.time()
s_mon = SpikeMonitor(P)
run(.1 * second)
duration = time.time() - start_time

print duration, len(s_mon)
plt.plot(s_mon.t/ms, s_mon.i, '.')
plt.show()
