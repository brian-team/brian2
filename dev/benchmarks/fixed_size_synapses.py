import time
import itertools
from collections import defaultdict
from processify import processify

from joblib import Memory
memory = Memory(location='.', verbose=0)

@processify
@memory.cache
def run_sim(target, n_select, n_total, fixed, trial):
    from brian2 import (set_device, device, NeuronGroup, Synapses, run,
                        clear_cache, prefs, second)
    assert n_select <= n_total
    if target == 'cpp_standalone':
        set_device('cpp_standalone')
        device.reinit()
        device.activate(directory=None)
    else:
        set_device('runtime')
        if target == 'cython':
            clear_cache(target)

        prefs.codegen.target = target
    source_neurons = NeuronGroup(1, '')
    target_neurons = NeuronGroup(n_total, '')
    syn = Synapses(source_neurons, target_neurons)
    start = time.time()
    if fixed:
        syn.connect(j='k for k in sample(N_post, size=n_select)')
    else:
        p = n_select / n_total
        syn.connect(j='k for k in sample(N_post, p=p)')
    if target == 'cpp_standalone':
        run(0*second)
    end = time.time()
    if fixed:
        assert len(syn) == n_select
    return end - start


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    total_sizes = [1_000, 10_000, 100_000, 1_000_000]
    select_sizes = [0, 5, 10, 50, 100, 500, 1_000, 5_000, 10_000]
    n_trials = 5
    targets = ['numpy', 'cython', 'cpp_standalone']
    fig, axes = plt.subplots(2, 3, sharey='row')
    for total_size, ax in zip(total_sizes, axes.flat):
        results_random = defaultdict(list)
        results_fixed = defaultdict(list)
        for select_size in select_sizes:
            if select_size > total_size:
                break
            for target in targets:
                trial_times_random = []
                trial_times_fixed = []
                for trial in range(n_trials):
                    t = run_sim(target, select_size, total_size, False, trial)
                    trial_times_random.append(t)
                    t = run_sim(target, select_size, total_size, True, trial)
                    trial_times_fixed.append(t)
                results_random[target].append(np.mean(trial_times_random))
                results_fixed[target].append(np.mean(trial_times_fixed))
        import pprint
        print(total_size)
        pprint.pprint(results_random)
        pprint.pprint(results_fixed)
        for idx, target in enumerate(targets):
            ax.plot(select_sizes[1:len(results_random[target])],
                    np.array(results_random[target][1:]) - results_random[target][0], 'o-', label=f'{target} (random)',
                    color=f'C{idx}', linestyle=':')
            ax.plot(select_sizes[1:len(results_fixed[target])],
                    np.array(results_fixed[target][1:]) - results_fixed[target][0], 'o-', label=f'{target} (fixed)',
                    color=f'C{idx}')
            ax.set(title=f'Total size: {total_size}', xlabel='selected',
                   ylabel='time (s)')
    axes[-1, -1].axis('off')
    for idx, target in enumerate(targets):
        axes[-1, -1].plot([], [], 'o-', label=f'{target} (random)',
                          color=f'C{idx}', linestyle=':')
        axes[-1, -1].plot([], [], 'o-', label=f'{target} (fixed)',
                          color=f'C{idx}')
    axes[-1, -1].legend()
    fig.tight_layout()
    fig.savefig('synapse_creation_benchmark.png')
