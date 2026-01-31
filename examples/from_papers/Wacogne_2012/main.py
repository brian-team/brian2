"""
MMN Simulation - Main Entry Point
=================================

This script acts as the primary configuration and execution hub for the project.
It defines the high-level parameters for the model and the experiment, then launches 
the simulation by calling the modular logic in `src/simulation.py`.

Usage:
    Run this script directly to execute the simulation:
    $ python main.py

Configuration:
    - 'model_params': Dictionary defining neuron counts, equations constants, and weight limits.
    - 'experiment_to_run': String selector to choose the experimental protocol 
      (e.g., 'classic', 'alternating', 'figure4_multi').

Output:
    - Generates interactive matplotlib figures.
    - Saves figure files to the 'fig_out/' directory.
"""

import os
import time
import matplotlib.pyplot as plt
from brian2 import ms, mV

from src.simulation import run_single_simulation
from src.plotting import create_figure4_multi_probability

if __name__ == '__main__':
    model_params = {
        'N_EXC': 40, 'N_INH': 40, 'N_E_MEM': 400, 'N_I_MEM': 100,
        'exc': {'a': 0.02, 'b': '0.2 + 0.04 * rand()**2', 'c': '(-65 + 10 * rand()**2)*mV',
                'd': '(8 - 2 * rand()**2)*mV', 'V_E': 40 * mV, 'V_I': -80 * mV, 'g_ampa': 0.0075, 'g_gaba': 0.0075,
                'tau_ampa': 2 * ms, 'tau_gaba': 10 * ms, 'g_nmda': 0.002, 'tau_nmda_rise': 2 * ms,
                'tau_nmda_decay': 100 * ms, 'alpha_nmda': 0.5 / ms, 'Mg2_conc': 0.001},
        'inh': {'a': '0.06 + 0.04 * rand()**2', 'b': 0.2, 'c': -65 * mV, 'd': 2 * mV, 'V_E': 40 * mV, 'g_ampa': 0.0075,
                'tau_ampa': 2 * ms, 'g_nmda': 0.002, 'tau_nmda_rise': 2 * ms, 'tau_nmda_decay': 100 * ms,
                'alpha_nmda': 0.5 / ms, 'Mg2_conc': 0.001},
        'input': {'a': 0.02, 'b': '0.2 + 0.04 * rand()**2', 'c': '(-65 + 10 * rand()**2)*mV',
                  'd': '(8 - 2 * rand()**2)*mV', 'sigma_noise': 0.5},
        'syn_weights': {
            "w_EE": 'clip(1.4 + sqrt(0.2 * 1.4) * randn(), 0, 5.0)',
            "w_EI": 'clip(4.5 + sqrt(0.2 * 4.5) * randn(), 0, 10.0)',
            "w_IE": 'clip(22.0 + sqrt(0.2 * 22.0) * randn(), 0, 35.0)'
        },
        'mem_all': {
            'exc': {'a': 0.02, 'b': '0.2 + 0.04*rand()**2', 'c': '(-65 + 10*rand()**2)*mV',
                    'd': '(15 - 3*rand()**2)*mV', 'V_E': 40 * mV, 'V_I': -80 * mV, 'g_ampa': 0.0075, 'g_gaba': 0.0075,
                    'tau_ampa': 1.25 * ms, 'tau_gaba': 14 * ms, 'g_nmda': 0.0001, 'tau_nmda_rise': 2 * ms,
                    'tau_nmda_decay': 80 * ms, 'alpha_nmda': 0.15 / ms, 'Mg2_conc': 0.001, 'sigma_noise': 0.10},
            'inh': {'a': '0.06 + 0.04*rand()**2', 'b': 0.2, 'c': -60 * mV, 'd': 10 * mV, 'V_E': 40 * mV,
                    'g_ampa': 0.0075, 'tau_ampa': 2 * ms, 'g_nmda': 0.0025, 'tau_nmda_rise': 4 * ms,
                    'tau_nmda_decay': 40 * ms, 'alpha_nmda': 0.15 / ms, 'Mg2_conc': 0.01, 'sigma_noise': 0.10},
            'weights': {"w_EE_mem": 140, "w_IE_mem": 0.5, "w_EI_mem": 40, "p_IE": 0.30, "p_EI": 0.2,
                        "CHAIN_DELAY": 1 * ms, "E_TO_I_DELAY": 0.5 * ms, "I_TO_E_DELAY": 0.5 * ms}
        }
    }

    classic_params = {'total_tones': 500, 'deviant_prob': 0.2, 'soa': 200 * ms, 'min_deviant_ms': 1000*ms}
    alternating_params = {'total_tones': 300, 'deviant_prob': 0.15, 'soa': 200 * ms, 'min_deviant_ms': 30000*ms}
    local_global_params = {'num_sequences': 100, 'intra_isi': 150 * ms, 'inter_soa': 1200 * ms, 'probabilities': [0.7, 0.2, 0.1]}
    omission_params = {'num_pairs': 1500, 'omission_prob': 0.10, 'isi': 200 * ms}

    experiment_to_run = 'classic' # 'classic', 'alternating', 'local_global', 'omission', 'figure4_multi'

    if experiment_to_run == 'figure4_multi':
        deviant_probs = [0.05, 0.10, 0.20, 0.30]
        results_list = []
        for prob_idx, prob in enumerate(deviant_probs):
            params = {'total_tones': 300, 'deviant_prob': prob, 'soa': 200 * ms, 'min_deviant_ms': 0 * ms}
            res, _ = run_single_simulation('classic', params, model_params, 2.4, 42 + prob_idx)
            results_list.append(res); plt.close('all')
        create_figure4_multi_probability(results_list, dt_max_ms=400, chain_delay_ms=2.0)
    elif experiment_to_run == 'classic':
        res, widgets = run_single_simulation('classic', classic_params, model_params, 2.4, 42)
    elif experiment_to_run == 'alternating':
        res, widgets = run_single_simulation('alternating', alternating_params, model_params, 3.0, 42)
    elif experiment_to_run == 'local_global':
        res, widgets = run_single_simulation('local_global', local_global_params, model_params, 3.0, 42)
    elif experiment_to_run == 'omission':
        res, widgets = run_single_simulation('omission', omission_params, model_params, 1.9, 42)

    outdir = os.path.join("fig_out", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    for i, num in enumerate(plt.get_fignums(), start=1):
        plt.figure(num).savefig(os.path.join(outdir, f"fig_{i:02d}.png"), dpi=200, bbox_inches='tight')
    print("Figures saved to:", outdir)
    plt.show()