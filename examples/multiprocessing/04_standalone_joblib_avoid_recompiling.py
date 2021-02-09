'''
Parallel processes using standalone mode

This example use multiprocessing to run the simulation.
The c++ code is generator and then is compiled.

The generated code is stored in `standalone{pid}` directory, which pid is the id of each processors.


`set_device` should be in the `run_sim` function.
This line initialises the C++ Standalone Device that keeps track of wether your network has been build or run before.

By moving the set_device line into the parallelised function, it create one C++ Standalone Device per process.
The `device.reinit()` need to be called` if you are running multiple simulations per process (there is 10 tau values and num_proc = 4).

Each simulation uses it's own code folder to generate the code for the simulation. This is controlled by the directory keyword to your set_device call. By setting `directory=None`, a temporary folder with random name is created in your `/tmp/` directory (at least on linux) to generate your code. This way, each simulation uses a different folder for code generation and there is nothing shared between the parallel processes.

If you don't set the directory argument, it defaults to `directory="output"`. In that case each process would use the same files to try to generate and compile your simulation, which would lead to compile/execution errors.

Actually, the way of setting the `directory=f"standalone{pid}"` is even better than using `directory=None` in this case. That is, giving each parallel thread *it's own directory to work on*. This way you avoid the problem of multiple threads working on the same code directories. But you also don't need to recompile the entire project at each simulation. What happens is that in the generated code in two consecutive simulations in a single thread will only differ in slightly (in your case only the tau parameter). The compiler will therefore only recompile the file that has changed and not the entire project (and there might be some other caching going on as well, don't know the internals). Just make sure that you don’t set clean=True, otherwise everything is recompiled from scratch each time.

The `numb_proc` set the number of processors.
`run_sim` is just a toy example that create a single neuron and connect a  `StateMonitor` to record the voltage.

for the discussios look at [here](https://brian.discourse.group/t/multiprocessing-in-standalone-mode/142/2)

'''

from joblib import Parallel, delayed
from time import time as wall_time
from os import system
from brian2 import *
import os

def clean_directories():
    system("rm -rf standalone*")


def run_sim(tau):
    
    pid = os.getpid()
    directory=f"standalone{pid}"
    set_device('cpp_standalone', directory=directory)
    print(f'RUNNING {pid}')

    if activate_standalone_mode:
        get_device().reinit()
        get_device().activate(
        # build_on_run=False,
        directory=directory)

    G = NeuronGroup(1, 'dv/dt = -v/tau : 1', method='euler')
    G.v = 1

    mon = StateMonitor(G, 'v', record=0)
    net = Network()
    net.add(G, mon)
    net.run(100 * ms)
    res = (mon.t/ms, mon.v[0])

    device.reinit()

    print(f'FINISHED {pid}')
    return res


if __name__ == "__main__":
    
    statr_time = wall_time()
    
    activate_standalone_mode = True
    
    n_jobs = 4
    tau_values = np.arange(10)*ms + 5*ms

    results = Parallel(n_jobs=n_jobs)(map(delayed(run_sim), tau_values))
    print(len(results), len(results[0]), results[0][1].shape)

    print("Done in {:10.3f}".format(wall_time() - statr_time))

    for tau_value, (t, v) in zip(tau_values, results):
        plt.plot(t, v, label=str(tau_value))
    plt.legend()
    plt.show()

    clean_directories()
