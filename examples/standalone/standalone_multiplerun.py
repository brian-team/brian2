'''
This example shows how to use multiple run in standalone mode and necessarily is not the optimal choise.
The example come from Tutorial part 3.
for a discussion look here
https://brian.discourse.group/t/multiple-run-in-standalone-mode/131
'''

import numpy as np
import pylab as plt
import brian2 as b2
from time import time

standalone_mode = True
directory_name = "output"


if standalone_mode:
    b2.set_device('cpp_standalone',
                  build_on_run=False,
                  ditectory=directory_name)


def simulate(tau):
    b2.start_scope()

    if standalone_mode:
        b2.get_device().reinit()
        b2.get_device().activate(build_on_run=False,
                                 directory=directory_name)

    eqs = '''
    dv/dt = -v/tau : 1
    '''

    net = b2.Network()
    P = b2.PoissonGroup(num_inputs, rates=input_rate)
    G = b2.NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='euler')
    S = b2.Synapses(P, G, on_pre='v += weight')
    S.connect()
    M = b2.SpikeMonitor(G)
    net.add(P)
    net.add(G)
    net.add(S)
    net.add(M)

    net.run(1000 * b2.ms)
    
    if standalone_mode:
        b2.get_device().build(directory=directory_name,
                              compile=True,
                              run=True,
                              debug=False)

    return M


if __name__ == "__main__":

    start_time = time()
    num_inputs = 100
    input_rate = 10 * b2.Hz
    weight = 0.1
    npoints = 15
    tau_range = np.linspace(1, 15, npoints) * b2.ms    

    output_rates = np.zeros(npoints)
    for ii in range(npoints):
        tau_i = tau_range[ii]
        M = simulate(tau_i)
        output_rates[ii] = M.num_spikes / b2.second

    print("Done in {}".format(time()-start_time))

    plt.plot(tau_range/b2.ms, output_rates)
    plt.xlabel(r'$\tau$ (ms)')
    plt.ylabel('Firing rate (sp/s)')
    plt.show()

# expected time to run the script:
# 53 s
