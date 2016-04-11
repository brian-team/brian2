from brian2 import *

BrianLogger.log_level_debug()
#brian_prefs.codegen.target = 'weave'

H = NeuronGroup(1, 'v:1', threshold='v>-1')

tau = 1*ms
eqs = '''
dv/dt = (el-v)/tau : 1 (unless refractory)
dx/dt = 0/tau : 1 (unless refractory)
dy/dt = 0/tau : 1
el : 1
'''
reset = '''
v = 0
x -= 0.05
y -= 0.05
'''
G = NeuronGroup(2, eqs, threshold='v>1', reset=reset, refractory=1*ms)
G.el = [2, 0]

Sx = Synapses(H, G, on_pre='x += dt*100*Hz')
Sx.connect(True)

Sy = Synapses(H, G, on_pre='y += dt*100*Hz')
Sy.connect(True)

M = StateMonitor(G, variables=True, record=True)

run(10*ms)

plot(M.t, M.v[0], label='v0')
plot(M.t, M.x[0], label='x0')
plot(M.t, M.y[0], label='y0')
plot(M.t, M.v[1], label='v1')
plot(M.t, M.x[1], label='x1')
plot(M.t, M.y[1], label='y1')
legend(loc='best')
ylim(-0.1, 1.1)
show()
