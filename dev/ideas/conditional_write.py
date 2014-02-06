from brian2 import *

#BrianLogger.log_level_debug()

H = NeuronGroup(1, 'v:1', threshold='v>-1')

tau = 1*ms
eqs = '''
dv/dt = (2-v)/tau : 1 (unless refractory)
dx/dt = 0/tau : 1 (unless refractory)
dy/dt = 0/tau : 1
'''
reset = '''
v = 0
x -= 0.05
y -= 0.05
'''
G = NeuronGroup(1, eqs, threshold='v>1', reset=reset, refractory=1*ms)

Sx = Synapses(H, G, pre='x += dt*100*Hz')
Sx.connect(True)

Sy = Synapses(H, G, pre='y += dt*100*Hz')
Sy.connect(True)

M = StateMonitor(G, variables=True, record=True)

run(10*ms)

plot(M.t, M.v.T, label='v')
plot(M.t, M.x.T, label='x')
plot(M.t, M.y.T, label='y')
legend(loc='best')
ylim(-0.1, 1.1)
show()
