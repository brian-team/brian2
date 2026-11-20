"""
Simulation Logic Module
=======================

This module contains the core logic for running the MMN experiments. It orchestrates
the creation of the network, the generation of stimulus paradigms, and the execution
of the simulation using Brian2.

Key Components:
    - run_single_simulation: The main entry point. Sets up the network, monitors, 
      and runs the simulation for a specific paradigm.
    - create_paradigm_sequence: Generates the tone sequences (frequency/timing) for different 
      experimental protocols (Classic, Alternating, Local-Global, Omission).

Experimental Paradigms:
    - 'classic': Standard oddball (AAAA B AAAA...).
    - 'alternating': Alternating tones (A B A B...).
    - 'local_global': Sequences establishing local/global rules (e.g. AAAA B vs AAAA A).
    - 'omission': Stimulus omission detection (AA AA AA A_).

Note:
    This module handles the 'instability_detector' to safely stop divergent simulations.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

from src.network import (
    create_neuron_group, create_cortical_column, create_memory_module,
    create_simple_memory_module, create_synaptic_connection, create_stdp_synapse
)
from src.analysis import (
    print_simulation_summary, analyze_weight_changes, report_missed
)
from src.plotting import (
    plot_weight_statistics, plot_example_synapses, plot_AB_sequence_window,
    create_mmn_comparison_plot_short, create_interactive_explorer,
    create_weight_profile_figure, plot_figure4_style, plot_input_vs_memory,
    create_figure2_interactive
)

def _ensure_no_consecutive_deviants(tones_array):
    """
    Ensures no two deviants (1s) are consecutive.
    """
    n_standards = np.count_nonzero(tones_array == 0)
    n_deviants = np.count_nonzero(tones_array == 1)

    if n_deviants > n_standards + 1:
        raise ValueError(f"{n_deviants} deviants cannot be placed among {n_standards} standards without adjacency.")

    result = np.zeros(n_standards, dtype=int)
    possible_indices = np.arange(n_standards + 1)
    chosen_indices = np.random.choice(possible_indices, n_deviants, replace=False)
    chosen_indices.sort()
    
    for index in chosen_indices[::-1]:
        result = np.insert(result, index, 1)

    return result

def create_paradigm_sequence(paradigm_name, params):
    """
    Generates the sequence of tones (stimuli) and their timing based on the selected paradigm.

    Paradigm Logic:
    - 'classic': Fixed probability of deviants (e.g., 20%). Ensures no consecutive deviants.
    - 'alternating': Tones alternate A-B-A-B. Deviants replace some B's with A's.
    - 'local_global': Uses 5-tone sequences (e.g. AAAA B vs AAAA A) to separate local vs global rules.
    - 'omission': Blocks of 'AA' (stimulus) vs 'A_' (omission).

    Args:
        paradigm_name: String identifier for the experiment type.
        params: Dictionary containing 'deviant_prob', 'total_tones', 'soa', etc.

    Returns:
        tones: NumPy array of tone identities (0 for Standard, 1 for Deviant).
        times: Brian2 Quantity object containing stimulus onset times.
    """
    print(f"Creating sequence for '{paradigm_name}'...")
    tones, times = None, None

    if paradigm_name == 'classic':
        total_tones = params['total_tones']
        deviant_prob = params['deviant_prob']
        soa = params['soa']
        min_dev_ms = params.get('min_deviant_ms', 0 * ms)

        n_deviants = int(total_tones * deviant_prob)
        n_standards = total_tones - n_deviants

        min_dev_idx = int(np.ceil((min_dev_ms / soa))) if min_dev_ms > 0 * ms else 0
        if min_dev_idx < 0: min_dev_idx = 0
        if min_dev_idx > total_tones: min_dev_idx = total_tones

        rem_standards = n_standards - min_dev_idx
        if rem_standards < 0:
            raise ValueError(f"min_deviant_ms too large.")
        if n_deviants > rem_standards + 1:
            raise ValueError("Too many deviants for the remaining slots.")

        prefix = np.zeros(min_dev_idx, dtype=int)
        segment_base = np.array([0] * rem_standards + [1] * n_deviants)
        segment_tones = _ensure_no_consecutive_deviants(segment_base)
        tones = np.concatenate([prefix, segment_tones])
        times = np.arange(total_tones) * soa

    elif paradigm_name == 'alternating':
        total_tones = params['total_tones']
        deviant_prob = params['deviant_prob']
        soa = params['soa']
        min_dev_ms = params.get('min_deviant_ms', 0 * ms)

        tones = np.tile([0, 1], total_tones // 2)
        if total_tones % 2 != 0: tones = np.append(tones, 0)

        if 0 < deviant_prob < 1.0:
            b_indices = np.where(tones == 1)[0]
            min_dev_idx = int(np.ceil((min_dev_ms / soa))) if min_dev_ms > 0 * ms else 0
            allowed = b_indices[b_indices >= min_dev_idx]
            num_to_change = int(len(allowed) * deviant_prob)
            if num_to_change > 0 and len(allowed) > 0:
                indices_to_change = np.random.choice(allowed, size=num_to_change, replace=False)
                tones[indices_to_change] = 0

        times = np.arange(len(tones)) * soa

    elif paradigm_name == 'local_global':
        num_sequences = params['num_sequences']
        intra_isi = params['intra_isi']
        inter_soa = params['inter_soa']
        probabilities = params['probabilities']
        sequence_map = {'standard': 'AAAAB', 'deviant': 'AAAAA', 'omission': 'AAAA'}
        chosen_sequences = np.random.choice(list(sequence_map.keys()), num_sequences, p=probabilities)
        full_tone_string = "".join([sequence_map[s] for s in chosen_sequences])
        tones = np.array([0 if T == 'A' else 1 for T in full_tone_string])
        time_list = []
        current_time = 0 * ms
        for seq_type in chosen_sequences:
            seq_length = len(sequence_map[seq_type])
            times_in_seq = current_time + np.arange(seq_length) * intra_isi
            time_list.extend(times_in_seq)
            current_time += (seq_length - 1) * intra_isi + inter_soa
        times = Quantity(time_list)

    elif paradigm_name == 'omission':
        num_pairs = params['num_pairs']
        omission_prob = params['omission_prob']
        isi = params['isi']
        num_omissions = int(num_pairs * omission_prob)
        num_doubles = num_pairs - num_omissions
        blocks = ['AA'] * num_doubles + ['A'] * num_omissions
        np.random.shuffle(blocks)
        tone_list, time_list = [], []
        current_time = 0 * ms
        for block in blocks:
            if block == 'AA':
                tone_list.extend([0])
                time_list.extend([current_time])
            else:
                tone_list.extend([1])
                time_list.extend([current_time])
            current_time += 2 * isi
        tones = np.array(tone_list)
        times = Quantity(time_list)

    else:
        raise ValueError(f"Unknown paradigm: {paradigm_name}")

    return tones, times

def run_single_simulation(paradigm_name, paradigm_params, model_params, stimulus_amplitude, seed_value):
    all_interactive_widgets = {}
    print("\n" + "#" * 70)
    print(f"### RUNNING SIMULATION: {paradigm_name.upper()} (SEED: {seed_value}) ###")
    print("#" * 70 + "\n")

    temp_dir = 'D:/brian2_temp' if os.name == 'nt' else None # Adjust for OS if needed, strictly using D: per user code
    if temp_dir:
        set_device('cpp_standalone', directory=temp_dir)
    else:
        set_device('runtime')

    start_scope()
    dt = 0.05 * ms
    defaultclock.dt = dt
    seed(seed_value)

    N_input_per_tone = 40
    N_input_total = N_input_per_tone * 2
    STIMULUS_DURATION = 10 * ms
    N_EXC = model_params['N_EXC']
    N_INH = model_params['N_INH']
    N_E_MEM = model_params['N_E_MEM']
    N_I_MEM = model_params['N_I_MEM']

    tones, times = create_paradigm_sequence(paradigm_name, paradigm_params)
    soa_or_isi = paradigm_params.get('soa', paradigm_params.get('isi', 200 * ms))
    total_duration = times[-1] + soa_or_isi * 2
    print(f"Simulation duration: {total_duration}")

    stimulus_dt = 1 * ms
    total_duration_steps = int(total_duration / stimulus_dt)
    current_arr = np.zeros((total_duration_steps, N_input_total))

    for t_stim, tone_type in zip(times, tones):
        start_idx = int(t_stim / stimulus_dt)
        end_idx = int((t_stim + STIMULUS_DURATION) / stimulus_dt)
        neuron_start = int(tone_type * N_input_per_tone)
        neuron_end = neuron_start + N_input_per_tone
        if end_idx < total_duration_steps:
            current_arr[start_idx:end_idx, neuron_start:neuron_end] = stimulus_amplitude

    stimulus_current = TimedArray(current_arr * mV, dt=stimulus_dt, name='stimulus_current')

    gate_dt = stimulus_dt
    gateA = np.zeros(total_duration_steps)
    gateB = np.zeros(total_duration_steps)
    gate_ALL = np.zeros(total_duration_steps)
    GATE_PRE = 5 * ms
    GATE_WIN = STIMULUS_DURATION + 20 * ms
    for t_stim, tone_type in zip(times, tones):
        start = max(0, int((t_stim - GATE_PRE) / gate_dt))
        end = min(total_duration_steps, int((t_stim + GATE_WIN) / gate_dt))
        gate_ALL[start:end] = 1.0
        if tone_type == 0:
            gateA[start:end] = 1.0
        else:
            gateB[start:end] = 1.0

    tone_gate_A = TimedArray(gateA, dt=gate_dt, name='tone_gate_A')
    tone_gate_B = TimedArray(gateB, dt=gate_dt, name='tone_gate_B')
    tone_gate_ALL = TimedArray(gate_ALL, dt=gate_dt, name='tone_gate_ALL')

    times_A_evt = times[tones == 0] + 1 * ms
    times_B_evt = times[tones == 1] + 1 * ms
    tone_trig_A = SpikeGeneratorGroup(1, indices=np.zeros(len(times_A_evt), dtype=int), times=times_A_evt, name='tone_trig_A')
    tone_trig_B = SpikeGeneratorGroup(1, indices=np.zeros(len(times_B_evt), dtype=int), times=times_B_evt, name='tone_trig_B')

    ThalamicInput = create_neuron_group(N_input_total, 'ThalamicInput', 'input', model_params['input'])
    column_A = create_cortical_column('A', N_EXC, N_INH, model_params['exc'], model_params['inh'], model_params['syn_weights'], record_states=True)
    column_B = create_cortical_column('B', N_EXC, N_INH, model_params['exc'], model_params['inh'], model_params['syn_weights'], record_states=True)

    neurons_rec = range(min(10, N_EXC))
    column_A['curr_mon_p'] = StateMonitor(column_A['P'], ['I_ampa', 'I_nmda', 'I_gaba'], record=neurons_rec, dt=2*ms, name='mon_curr_A_P')
    column_A['curr_mon_pe'] = StateMonitor(column_A['PE'], ['I_ampa', 'I_nmda', 'I_gaba'], record=neurons_rec, dt=2*ms, name='mon_curr_A_PE')
    column_B['curr_mon_p'] = StateMonitor(column_B['P'], ['I_ampa', 'I_nmda', 'I_gaba'], record=neurons_rec, dt=2*ms, name='mon_curr_B_P')
    column_B['curr_mon_pe'] = StateMonitor(column_B['PE'], ['I_ampa', 'I_nmda', 'I_gaba'], record=neurons_rec, dt=2*ms, name='mon_curr_B_PE')

    SIMPLE_MEM_PARAMS = dict(theta=1.0, v_reset=0.0, v_rest=0.0, tau_m=5 * ms, t_ref=1 * ms, J_ff=1.1, d_ff=1.5 * ms, J_in=2.4, tau_gate=100 * ms)
    if True: # Always use simple memory for now as per reference
        memory_module_A = create_simple_memory_module('A', SIMPLE_MEM_PARAMS, N_E_MEM)
        memory_module_B = create_simple_memory_module('B', SIMPLE_MEM_PARAMS, N_E_MEM)
    
    thalamic_statemon = StateMonitor(ThalamicInput, 'v', record=range(min(10, N_input_total)), dt=1 * ms, name='statemon_thalamic')

    syn_Thalamic_PE_A = create_synaptic_connection(ThalamicInput, column_A['PE'], 0.9, model_params['syn_weights']["w_EE"],
                                                   's_ampa_post = s_ampa_post + w; x_nmda_post = x_nmda_post + w * 0.2',
                                                   delay_model='rand()*15*ms', cond=f'i < {N_input_per_tone}')
    syn_Thalamic_PE_B = create_synaptic_connection(ThalamicInput, column_B['PE'], 0.9, model_params['syn_weights']["w_EE"],
                                                   's_ampa_post = s_ampa_post + w; x_nmda_post = x_nmda_post + w * 0.2',
                                                   delay_model='rand()*15*ms', cond=f'i >= {N_input_per_tone}')

    U = 1.0 # J_in and U are from SIMPLE_MEM_PARAMS logic
    trig_code = f'''v_post = v_post + w*{SIMPLE_MEM_PARAMS['J_in']}*x_gate_post*tone_gate_A(t)
                    x_gate_post = x_gate_post - {U}*x_gate_post*tone_gate_A(t)*w'''
    syn_P_to_Mem_A = Synapses(column_A['P'], memory_module_A['E_chain'], model='w:1', on_pre=trig_code, name='trig_A')
    syn_P_to_Mem_A.connect(condition='j==0'); syn_P_to_Mem_A.w=1.0; syn_P_to_Mem_A.delay='rand()*15*ms'

    trig_code_B = trig_code.replace('tone_gate_A', 'tone_gate_B')
    syn_P_to_Mem_B = Synapses(column_B['P'], memory_module_B['E_chain'], model='w:1', on_pre=trig_code_B, name='trig_B')
    syn_P_to_Mem_B.connect(condition='j==0'); syn_P_to_Mem_B.w=1.0; syn_P_to_Mem_B.delay='rand()*15*ms'

    # Backups triggers
    syn_Tone_to_Mem_A = Synapses(tone_trig_A, memory_module_A['E_chain'], model='w:1', on_pre=f'v_post=v_post+w*{SIMPLE_MEM_PARAMS["J_in"]}', name='trig_tone_A')
    syn_Tone_to_Mem_A.connect(condition='j==0'); syn_Tone_to_Mem_A.w=1.0; syn_Tone_to_Mem_A.delay=50*ms
    
    syn_Tone_to_Mem_B = Synapses(tone_trig_B, memory_module_B['E_chain'], model='w:1', on_pre=f'v_post=v_post+w*{SIMPLE_MEM_PARAMS["J_in"]}', name='trig_tone_B')
    syn_Tone_to_Mem_B.connect(condition='j==0'); syn_Tone_to_Mem_B.w=1.0; syn_Tone_to_Mem_B.delay=50*ms

    # Restore missing Thalamic -> Memory connections
    syn_Thal_to_Mem_A = Synapses(
        ThalamicInput, memory_module_A['E_chain'], model='w:1',
        on_pre=f'''
                v_post      = v_post      + w*{SIMPLE_MEM_PARAMS['J_in']}*x_gate_post*tone_gate_A(t)
                x_gate_post = x_gate_post - {U}*x_gate_post*tone_gate_A(t)*w
            ''',
        name='trig_thal_A'
    )
    syn_Thal_to_Mem_A.connect(condition=f'(i < {N_input_per_tone}) and (j==0)')
    syn_Thal_to_Mem_A.w = 1.0
    syn_Thal_to_Mem_A.delay = 50 * ms

    syn_Thal_to_Mem_B = Synapses(
        ThalamicInput, memory_module_B['E_chain'],
        model='w : 1',
        on_pre=f'''
        v_post = v_post + w * {SIMPLE_MEM_PARAMS["J_in"]} * x_gate_post * tone_gate_B(t)
        x_gate_post = x_gate_post - {U} * x_gate_post * tone_gate_B(t)
        ''',
        name='Thal_to_Mem_B'
    )
    syn_Thal_to_Mem_B.connect(condition=f'(i >= {N_input_per_tone}) and (j == 0)')
    syn_Thal_to_Mem_B.w = 1.0

    # Plastic connections
    initial_weights = {}
    synapse_map = {}
    conn_prob = 0.5
    
    def create_conn(Ns, Nt, p): return {'i': np.random.randint(0, Ns, int(Ns*Nt*p)), 'j': np.random.randint(0, Nt, int(Ns*Nt*p))}
    
    mu, sigma = 0.4, 0.08
    
    # A->A
    cAA = create_conn(N_E_MEM, N_EXC, conn_prob)
    wAA = np.clip(mu + sigma*np.random.randn(len(cAA['i'])), 0.01, 10.0)
    initial_weights['A_A'] = np.copy(wAA)
    syn_Mem_to_P_A = create_stdp_synapse(memory_module_A['E_chain'], column_A['P'], wAA, conn_data=cAA, 
                                         A_plus=0.03, A_minus=-0.04, taupre_ms=12.0, taupost_ms=24.0, name='stdp_AA')
    synapse_map['A_A'] = syn_Mem_to_P_A
    
    # B->B
    cBB = create_conn(N_E_MEM, N_EXC, conn_prob)
    wBB = np.clip(mu + sigma*np.random.randn(len(cBB['i'])), 0.01, 10.0)
    initial_weights['B_B'] = np.copy(wBB)
    syn_Mem_to_P_B = create_stdp_synapse(memory_module_B['E_chain'], column_B['P'], wBB, conn_data=cBB, 
                                         A_plus=0.03, A_minus=-0.04, taupre_ms=12.0, taupost_ms=24.0, name='stdp_BB')
    synapse_map['B_B'] = syn_Mem_to_P_B
    
    # A->B
    cAB = create_conn(N_E_MEM, N_EXC, conn_prob)
    wAB = np.clip(mu + sigma*np.random.randn(len(cAB['i'])), 0.01, 10.0)
    initial_weights['A_B'] = np.copy(wAB)
    syn_MemA_to_PB = create_stdp_synapse(memory_module_A['E_chain'], column_B['P'], wAB, conn_data=cAB, 
                                         A_plus=0.01, A_minus=-0.05, taupre_ms=12.0, taupost_ms=24.0, name='stdp_AB')
    synapse_map['A_B'] = syn_MemA_to_PB
    
    # B->A
    cBA = create_conn(N_E_MEM, N_EXC, conn_prob)
    wBA = np.clip(mu + sigma*np.random.randn(len(cBA['i'])), 0.01, 10.0)
    initial_weights['B_A'] = np.copy(wBA)
    syn_MemB_to_PA = create_stdp_synapse(memory_module_B['E_chain'], column_A['P'], wBA, conn_data=cBA, 
                                         A_plus=0.08, A_minus=-0.02, taupre_ms=12.0, taupost_ms=24.0, name='stdp_BA')
    synapse_map['B_A'] = syn_MemB_to_PA

    mon_weights_A_A = StateMonitor(syn_Mem_to_P_A, 'w', record=True, dt=100 * ms, name='mon_w_A_A')
    mon_weights_B_B = StateMonitor(syn_Mem_to_P_B, 'w', record=True, dt=100 * ms, name='mon_w_B_B')
    mon_weights_A_B = StateMonitor(syn_MemA_to_PB, 'w', record=True, dt=100 * ms, name='mon_w_A_B')
    mon_weights_B_A = StateMonitor(syn_MemB_to_PA, 'w', record=True, dt=100 * ms, name='mon_w_B_A')
    spikemon_input = SpikeMonitor(ThalamicInput, name='spikemon_input')

    @network_operation(dt=1 * ms, name='instability_detector')
    def instability_detector(t):
        groups = {**column_A, **column_B, **memory_module_A, **memory_module_B}
        for name, g in groups.items():
            if hasattr(g, 'v') and np.any(np.isnan(g.v)):
                print(f"!!! INSTABILITY at {t/ms}ms in {name} !!!")
                net.stop()
                return

    
    # Explicitly collect all objects to avoid dependency errors
    all_objects = [ThalamicInput, spikemon_input, thalamic_statemon,
                   syn_Thalamic_PE_A, syn_Thalamic_PE_B, syn_P_to_Mem_A, syn_P_to_Mem_B,
                   syn_Mem_to_P_A, syn_Mem_to_P_B, syn_MemA_to_PB, syn_MemB_to_PA,
                   mon_weights_A_A, mon_weights_B_B, mon_weights_A_B, mon_weights_B_A,
                   tone_trig_A, tone_trig_B, instability_detector]
                   
    all_objects.extend(column_A.values())
    all_objects.extend(column_B.values())
    
    for obj in memory_module_A.values():
        if obj is not None: all_objects.append(obj)
    for obj in memory_module_B.values():
        if obj is not None: all_objects.append(obj)

    net = Network(all_objects)
    print("\n>>> RUNNING SIMULATION...")
    net.run(total_duration, report='text')
    print(">>> SIMULATION COMPLETED.")

    def _safe_mon(mon): return mon if mon is not None else None

    all_spike_monitors = {
        'Input Thalamic': spikemon_input,
        'Memory A (E_chain)': memory_module_A['spikemon_mem_e'],
        'Memory B (E_chain)': memory_module_B['spikemon_mem_e'],
        'Column A - PE': column_A['spikemon_pe'], 'Column A - P': column_A['spikemon_p'], 'Column A - I': column_A['spikemon_i'],
        'Column B - PE': column_B['spikemon_pe'], 'Column B - P': column_B['spikemon_p'], 'Column B - I': column_B['spikemon_i'],
    }
    final_weights_dict = {k: m.w[:, -1] for k, m in [('A_A', mon_weights_A_A), ('B_B', mon_weights_B_B), ('A_B', mon_weights_A_B), ('B_A', mon_weights_B_A)]}

    print_simulation_summary(all_spike_monitors, final_weights_dict, N_input_per_tone)
    for k in ['A_A', 'B_B', 'A_B', 'B_A']:
        analyze_weight_changes(initial_weights[k], final_weights_dict[k], col_id=k)

    weight_stats = {k: {'t': m.t/ms, 'mean': np.mean(m.w, axis=0), 'min': np.min(m.w, axis=0), 'max': np.max(m.w, axis=0)} 
                    for k, m in [('A_A', mon_weights_A_A), ('B_B', mon_weights_B_B), ('A_B', mon_weights_A_B), ('B_A', mon_weights_B_A)]}
    plot_weight_statistics(weight_stats)

    plot_example_synapses(mon_weights_A_A, synapse_map['A_A'], initial_weights['A_A'], final_weights_dict['A_A'], 'A->A', 'royalblue')
    plot_example_synapses(mon_weights_A_B, synapse_map['A_B'], initial_weights['A_B'], final_weights_dict['A_B'], 'A->B', 'mediumseagreen')
    plot_example_synapses(mon_weights_B_A, synapse_map['B_A'], initial_weights['B_A'], final_weights_dict['B_A'], 'B->A', 'darkorange')
    plot_example_synapses(mon_weights_B_B, synapse_map['B_B'], initial_weights['B_B'], final_weights_dict['B_B'], 'B->B', 'crimson')

    result_package = {
        "tones": tones, "times": times, "total_duration": total_duration,
        "deviant_prob": paradigm_params.get('deviant_prob', 0.1),
        "synapse_map": synapse_map,
        "wmon_dict": {'A_A': mon_weights_A_A, 'A_B': mon_weights_A_B, 'B_A': mon_weights_B_A, 'B_B': mon_weights_B_B},
        "N_E_MEM": N_E_MEM,
        "monitors_A": {k: _safe_mon(v) for k, v in column_A.items()},
        "monitors_B": {k: _safe_mon(v) for k, v in column_B.items()},
        "memory_module_A": {k: _safe_mon(v) for k, v in memory_module_A.items()},
        "memory_module_B": {k: _safe_mon(v) for k, v in memory_module_B.items()},
        "thalamic_spikemon": _safe_mon(spikemon_input),
        "N_input_per_tone": N_input_per_tone,
        "final_weights": final_weights_dict
    }

    create_mmn_comparison_plot_short(tones, times, column_A, column_B, spikemon_input, thalamic_statemon, 
                                     memory_module_A, memory_module_B, N_input_per_tone)
    
    plot_AB_sequence_window(tones, times, column_A, column_B, spikemon_input, thalamic_statemon, N_input_per_tone, prefer='last')

    fig2, wid2 = create_figure2_interactive(result_package)
    if fig2: all_interactive_widgets['fig2'] = fig2

    all_interactive_widgets['explorer'] = create_interactive_explorer(
        total_duration, column_A, column_B, spikemon_input, N_input_per_tone, 
        memory_module_A, memory_module_B, model_params)
        
    all_interactive_widgets['weights'] = create_weight_profile_figure(
        total_duration, model_params, 
        syn_Mem_to_P_A, syn_MemA_to_PB, syn_Mem_to_P_B, syn_MemB_to_PA,
        mon_weights_A_A, mon_weights_A_B, mon_weights_B_B, mon_weights_B_A)

    all_interactive_widgets['fig4'] = plot_figure4_style(tones, times, synapse_map, result_package['wmon_dict'], 
                                                         memory_module_A['spikemon_mem_e'], memory_module_B['spikemon_mem_e'], 
                                                         N_E_MEM, chain_delay_ms=2.0)

    spk_mem_A = memory_module_A['spikemon_mem_e']
    spk_mem_B = memory_module_B['spikemon_mem_e']
    SOA_eff = paradigm_params.get('soa', 200*ms)
    chain_delay = model_params['mem_all']['weights']['CHAIN_DELAY']
    
    summary_A, rows_A = plot_input_vs_memory('A', spikemon_input, N_input_per_tone, spk_mem_A, tones, times, SOA_eff, chain_delay)
    report_missed(rows_A)
    summary_B, rows_B = plot_input_vs_memory('B', spikemon_input, N_input_per_tone, spk_mem_B, tones, times, SOA_eff, chain_delay)
    report_missed(rows_B)

    return result_package, all_interactive_widgets
