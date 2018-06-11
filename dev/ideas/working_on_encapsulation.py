from brian2 import *
from brian2.core.variables import Parameter
set_device('cpp_standalone', directory='encapsulation', build_on_run=True,
           simulation_class_name='encapsulation_sim')

# tau = Parameter(9*ms, name='tau')
# G = NeuronGroup(1, 'dv/dt=(2-v)/tau:1', threshold='v>1', reset='v=0')
# S = Synapses(G, G, on_pre='v += 0')
# S.connect()
# M = SpikeMonitor(G)
#
# run(100*ms)
#
# print M.t/ms

G1 = SpikeGeneratorGroup(1, [0], [1] * ms)
G2 = SpikeGeneratorGroup(1, [0], [2] * ms)
S = Synapses(G1, G2, '''pre_value : 1
                        post_value : 1''',
             pre='pre_value +=1',
             post='post_value +=2')
S.connect()
syn_mon = StateMonitor(S, ['pre_value', 'post_value'], record=[0],
                       when='end')
run(3 * ms)

#device.build(directory='encapsulation', run=False)

# all_t = []
# all_i = []
# for i, tau in enumerate(linspace(1, 50, 100)*ms):
#     device.run(directory='encapsulation', with_output=True, run_args=['-tau=%f' % float(tau)])
#     all_t.append(M.t[:])
#     all_i.append(i*ones(len(M.t)))
#
# all_t = hstack(all_t)
# all_i = hstack(all_i)
# plot(all_t/ms, all_i, '.k')
# show()
