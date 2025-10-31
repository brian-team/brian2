"""
This example shows how to run several, independent simulations in standalone mode using multiple processes to run the
simulations in parallel.
Given that this example only involves a single neuron, an alternative – and arguably more elegant – solution
would be to run the simulations in a single `NeuronGroup`, where each neuron receives input with a different rate.

The example is a standalone equivalent of the one presented in :doc:`/tutorials/3-intro-to-brian-simulations`.

Note that Python's `multiprocessing` module cannot deal with user-defined functions (including `TimedArray`) and other
complex code structures. If you run into `PicklingError` or `AttributeError` exceptions, you might
have to use the `pathos` (https://pypi.org/project/pathos) package instead, which can handle more complex
code structures.
"""
import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2
from time import time

b2.set_device('cpp_standalone', build_on_run=False)

class SimWrapper:
    def __init__(self):
        self.net = b2.Network()
        P = b2.PoissonGroup(num_inputs, rates=input_rate)
        eqs = """
            dv/dt = -v/tau : 1
            tau : second (constant)
            """
        G = b2.NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='euler', name='neuron')
        S = b2.Synapses(P, G, on_pre='v += weight')
        S.connect()
        M = b2.SpikeMonitor(G, name='spike_monitor')
        self.net.add([P, G, S, M])

        self.net.run(1000 * b2.ms)

        self.device = b2.get_device()
        self.device.build(run=False, directory=None)  # compile the code, but don't run it yet

    def do_run(self, tau_i):
        # Workaround to set the device globally in this context
        from brian2.devices import device_module
        device_module.active_device = self.device

        result_dir = f'result_{tau_i}'
        self.device.run(run_args={self.net['neuron'].tau: tau_i},
                        results_directory=result_dir)
        return self.net["spike_monitor"].num_spikes/ b2.second


if __name__ == "__main__":
    start_time = time()
    num_inputs = 100
    input_rate = 10 * b2.Hz
    weight = 0.1

    npoints = 15
    tau_range = np.linspace(1, 15, npoints) * b2.ms

    sim = SimWrapper()

    from multiprocessing import Pool
    with Pool(npoints) as pool:
        output_rates = pool.map(sim.do_run, tau_range)

    print(f"Done in {time() - start_time}")

    plt.plot(tau_range/b2.ms, output_rates)
    plt.xlabel(r"$\tau$ (ms)")
    plt.ylabel("Firing rate (sp/s)")
    plt.show()
