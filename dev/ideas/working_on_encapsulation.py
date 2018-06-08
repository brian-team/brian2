from brian2 import *
from brian2.core.variables import Parameter
set_device('cpp_standalone', directory='encapsulation', build_on_run=False)

tau = Parameter(9*ms, name='tau')
G = NeuronGroup(1, 'dv/dt=(2-v)/tau:1', threshold='v>1', reset='v=0')
M = SpikeMonitor(G)

run(100*ms)

device.build(directory='encapsulation', run=False)

all_t = []
all_i = []
for i, tau in enumerate(linspace(1, 50, 100)*ms):
    device.run(directory='encapsulation', with_output=True, run_args=['-tau=%f' % float(tau)])
    all_t.append(M.t[:])
    all_i.append(i*ones(len(M.t)))

all_t = hstack(all_t)
all_i = hstack(all_i)
plot(all_t/ms, all_i, '.k')
show()
