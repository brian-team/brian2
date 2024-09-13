"""
Parallel processes using Cython

This example use multiprocessing to run several simulations in parallel.
The code is using the default runtime mode (and Cython compilation, if
possible).

The ``numb_proc`` variable set the number of processes. ``run_sim`` is just a
toy example that creates a single neuron and connects a `StateMonitor` to
record the voltage.

For more details see the `github issue 1154 <https://github.com/brian-team/brian2/issues/1154#issuecomment-582994117>`_:

Note that Python's `multiprocessing` module cannot deal with user-defined functions (including `TimedArray`) and other
complex code structures. If you run into `PicklingError` or `AttributeError` exceptions, you might
have to use the `pathos` (https://pypi.org/project/pathos) package instead, which can handle more complex
code structures.
"""
import os
import multiprocessing

from brian2 import *


def run_sim(tau):
    pid = os.getpid()
    print(f'RUNNING {pid}')
    G = NeuronGroup(1, 'dv/dt = -v/tau : 1', method='exact')
    G.v = 1
    mon = StateMonitor(G, 'v', record=0)
    run(100*ms)
    print(f'FINISHED {pid}')
    return mon.t/ms, mon.v[0]


if __name__ == "__main__":
    num_proc = 4

    tau_values = np.arange(10)*ms + 5*ms
    with multiprocessing.Pool(num_proc) as p:
        results = p.map(run_sim, tau_values)

    for tau_value, (t, v) in zip(tau_values, results):
        plt.plot(t, v, label=str(tau_value))
    plt.legend()
    plt.show()
