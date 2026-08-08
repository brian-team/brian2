"""
Data Analysis Module
====================

This module provides tools for quantifying the simulation results and detecting the Mismatch Negativity (MMN) effect. 
It processes the raw monitor data (spikes, weights) to generate statistical summaries.

Functions:
    - quantify_mmn_response: Calculates the difference in spike counts between deviant and standard responses.
    - analyze_weight_changes: Computes statistics on synaptic weight evolution (plasticity).
    - analyze_omission_response: Specifically detects responses to omitted stimuli.
    - print_simulation_summary: outputs a text-based summary of all key metrics to the console.
    - summarize_single_run: Helper to package scalar metrics and PSTH traces for a single run.
    - mem_start_success_ratio: Analyzes how effectively the memory chains were triggered.
"""

import numpy as np
from brian2 import ms, Quantity, SpikeMonitor

def _as_ms_quantity(x):
    """
    Ensure x is a Brian2 Quantity with unit ms.
    If x is already a Quantity, return it.
    If x is a number, assume it represents ms.
    """
    return x if isinstance(x, Quantity) else x * ms

def analyze_weight_changes(initial_weights, final_weights, col_id, threshold=0.1):
    """
    Compare initial and final weights to count how many synapses strengthened,
    weakened, or stayed the same.
    """
    valid_indices = np.isfinite(initial_weights) & np.isfinite(final_weights)
    initial_weights = initial_weights[valid_indices]
    final_weights = final_weights[valid_indices]

    total_synapses = len(initial_weights)
    if total_synapses == 0:
        print(f"No valid synapses found for analysis in Column {col_id}.")
        return

    change = final_weights - initial_weights

    strengthened = np.sum(change > threshold)
    weakened = np.sum(change < -threshold)
    unchanged = total_synapses - strengthened - weakened

    initial_mean = np.mean(initial_weights)
    final_mean = np.mean(final_weights)

    print(f"\n--- COLUMN {col_id} WEIGHT CHANGE ANALYSIS ---")
    print(f"Total Valid Synapses: {total_synapses}")
    print(f"Strengthened (> +{threshold}): {strengthened} ({strengthened / total_synapses:.1%})")
    print(f"Weakened (< -{threshold}): {weakened} ({weakened / total_synapses:.1%})")
    print(f"Significantly Unchanged: {unchanged} ({unchanged / total_synapses:.1%})")
    print(f"Mean Weight Change: {initial_mean:.3f} -> {final_mean:.3f}")
    print("-" * (36 + len(str(col_id))))

def print_simulation_summary(spike_monitors, final_weights, N_input_per_tone):
    """
    Print a general summary at the end of the simulation.
    Shows weight stats and spike counts for key groups.
    """
    print("\n" + "=" * 50)
    print(" " * 12 + "SIMULATION SUMMARY")
    print("=" * 50)

    # --- POST-LEARNING WEIGHT STATISTICS ---
    print("\n--- POST-LEARNING WEIGHT STATISTICS ---")
    w_aa = final_weights.get('A_A', np.array([0]))
    w_bb = final_weights.get('B_B', np.array([0]))
    w_ab = final_weights.get('A_B', np.array([0]))
    w_ba = final_weights.get('B_A', np.array([0]))

    print(f"Weights (A->A): Min={np.min(w_aa):.3f}, Max={np.max(w_aa):.3f}, Mean={np.mean(w_aa):.3f}")
    print(f"Weights (B->B): Min={np.min(w_bb):.3f}, Max={np.max(w_bb):.3f}, Mean={np.mean(w_bb):.3f}")
    print(f"Weights (A->B): Min={np.min(w_ab):.3f}, Max={np.max(w_ab):.3f}, Mean={np.mean(w_ab):.3f}")
    print(f"Weights (B->A): Min={np.min(w_ba):.3f}, Max={np.max(w_ba):.3f}, Mean={np.mean(w_ba):.3f}")

    # --- THALAMIC INPUT SPIKE COUNTS ---
    print("\n--- THALAMIC INPUT SPIKE COUNTS ---")
    thalamic_mon = spike_monitors.get('Input Thalamic')
    thalamic_a_count = 0
    thalamic_b_count = 0
    if thalamic_mon:
        thalamic_a_count = np.sum(thalamic_mon.i < N_input_per_tone)
        thalamic_b_count = np.sum(thalamic_mon.i >= N_input_per_tone)
    print(f"Thalamic Input A: {thalamic_a_count}")
    print(f"Thalamic Input B: {thalamic_b_count}")

    # --- CORTICAL COLUMN SPIKE COUNTS ---
    print("\n--- CORTICAL COLUMN SPIKE COUNTS ---")
    mon_pe_a = spike_monitors.get('Column A - PE')
    pe_a = mon_pe_a.num_spikes if mon_pe_a else 0
    mon_p_a = spike_monitors.get('Column A - P')
    p_a = mon_p_a.num_spikes if mon_p_a else 0
    mon_i_a = spike_monitors.get('Column A - I')
    i_a = mon_i_a.num_spikes if mon_i_a else 0
    print(f"Column A - PE: {pe_a}, P: {p_a}, I: {i_a}")

    mon_pe_b = spike_monitors.get('Column B - PE')
    pe_b = mon_pe_b.num_spikes if mon_pe_b else 0
    mon_p_b = spike_monitors.get('Column B - P')
    p_b = mon_p_b.num_spikes if mon_p_b else 0
    mon_i_b = spike_monitors.get('Column B - I')
    i_b = mon_i_b.num_spikes if mon_i_b else 0
    print(f"Column B - PE: {pe_b}, P: {p_b}, I: {i_b}")

    # --- MEMORY MODULE SPIKE COUNTS ---
    print("\n--- MEMORY MODULE SPIKE COUNTS ---")
    mon_mem_a = spike_monitors.get('Memory A (E_chain)')
    mem_a = mon_mem_a.num_spikes if mon_mem_a else 0
    mon_mem_b = spike_monitors.get('Memory B (E_chain)')
    mem_b = mon_mem_b.num_spikes if mon_mem_b else 0
    print(f"Memory A E_chain: {mem_a}")
    print(f"Memory B E_chain: {mem_b}")
    print("\n" + "=" * 50 + "\n")

def quantify_mmn_response(mon_pe_A, mon_pe_B, tones, times, window_start_ms=0, window_end_ms=150):
    """
    Quantifies MMN effect by counting PE neuron spikes in a specific time window.
    """
    print("\n" + "-" * 50)
    print("--- MMN EFFECT QUANTITATIVE ANALYSIS ---")

    t_standard_event = None
    t_deviant_event = None

    # Find the first instance where a standard tone (0) is followed by a deviant tone (1)
    for i in range(len(tones) - 1):
        if tones[i] == 0 and tones[i + 1] == 1:
            t_standard_event = times[i]
            t_deviant_event = times[i + 1]
            print(f"Found pair for analysis: Standard (t={t_standard_event / ms:.0f}ms), Deviant (t={t_deviant_event / ms:.0f}ms)")
            break

    if t_standard_event is None or t_deviant_event is None:
        print("WARNING: Could not find a suitable 'standard followed by deviant' pair for analysis.")
        print("-" * 50 + "\n")
        return

    start_window = window_start_ms * ms
    end_window = window_end_ms * ms

    # Count spikes
    std_mask = (mon_pe_A.t >= t_standard_event + start_window) & (mon_pe_A.t < t_standard_event + end_window)
    std_spike_count = len(mon_pe_A.t[std_mask])

    dev_mask = (mon_pe_B.t >= t_deviant_event + start_window) & (mon_pe_B.t < t_deviant_event + end_window)
    dev_spike_count = len(mon_pe_B.t[dev_mask])

    print(f"Analysis Window: {window_start_ms} ms to {window_end_ms} ms after stimulus")
    print(f"Response to Standard (PE_A Spikes): {std_spike_count}")
    print(f"Response to Deviant (PE_B Spikes):   {dev_spike_count}")

    if dev_spike_count > std_spike_count:
        print(">>> RESULT: MMN effect observed (Error response to deviant is higher).")
    else:
        print(">>> RESULT: Expected MMN effect NOT observed.")
    print("-" * 50 + "\n")

def _pick_AB_after(tones, times, t_min_ms=0, prefer='last'):
    """
    Find an A(0) -> B(1) consecutive pair after time t_min_ms.
    Returns: (t_A, t_B) in Brian2 ms units.
    """
    tones = np.asarray(tones)
    cand = np.where((tones[:-1] == 0) & (tones[1:] == 1))[0]  # B indices - 1
    if len(cand) == 0:
        return None, None
    
    tA_ms = np.array([float(_as_ms_quantity(times[i]) / ms) for i in cand])
    mask = (tA_ms >= t_min_ms)
    cand = cand[mask]
    if len(cand) == 0:
        return None, None
    i = cand[-1] if prefer == 'last' else cand[0]
    return _as_ms_quantity(times[i]), _as_ms_quantity(times[i + 1])

def _find_AB_pairs_after(tones, times, t_min_ms=0, gap_target_ms=200, gap_tol_ms=5):
    """
    Returns all A->B pairs after t_min_ms matching the gap constraint.
    Returns: list of (tA_ms, tB_ms)
    """
    tones = np.asarray(tones)
    idxA = np.where((tones[:-1] == 0) & (tones[1:] == 1))[0]
    if len(idxA) == 0:
        return []

    tA_ms = np.array([float(_as_ms_quantity(times[i]) / ms) for i in idxA])
    tB_ms = np.array([float(_as_ms_quantity(times[i + 1]) / ms) for i in idxA])

    m = tA_ms >= t_min_ms
    if gap_target_ms is not None:
        m &= np.abs((tB_ms - tA_ms) - gap_target_ms) <= gap_tol_ms

    idxA = idxA[m]
    if len(idxA) == 0:
        return []

    pairs = [(_as_ms_quantity(times[i]), _as_ms_quantity(times[i + 1])) for i in idxA]
    return pairs

def select_mmn_pair(tones, times, *, exclude_first=200, standard_tone=0, deviant_tone=1,
                    min_tail_ms=150, sim_end=None):
    """
    Selects a late deviant (AB) and the preceding standard (AA).
    Ensures 'min_tail_ms' remains after the event.
    Returns: (t_std_ms, t_dev_ms)
    """
    tones = np.asarray(tones)
    times = np.asarray(times)

    AB = np.where((tones[:-1] == standard_tone) & (tones[1:] == deviant_tone))[0] + 1
    AB = AB[AB > exclude_first]
    if len(AB) == 0:
        return None, None

    AA = np.where((tones[:-1] == standard_tone) & (tones[1:] == standard_tone))[0] + 1
    AA = AA[AA > exclude_first]

    if sim_end is None:
        sim_end = _as_ms_quantity(times[-1])
    else:
        sim_end = _as_ms_quantity(sim_end)

    for dev_idx in AB[::-1]:
        prev_AA = AA[AA < dev_idx]
        if len(prev_AA) == 0:
            continue
        t_dev = _as_ms_quantity(times[dev_idx])
        if t_dev + min_tail_ms * ms <= sim_end:
            t_std = _as_ms_quantity(times[prev_AA[-1]])
            return t_std, t_dev

    t_dev = _as_ms_quantity(times[AB[-1]])
    prev_AA = AA[AA < AB[-1]]
    t_std = _as_ms_quantity(times[prev_AA[-1]]) if len(prev_AA) else None
    return t_std, t_dev

def print_mmn_summary(
        tones, times,
        monitors_A, monitors_B,
        layers=('pe', 'p'),
        window_ms=(0, 150),
        baseline_ms=(-50, 0),
        exclude_first=200,
        n_trials=100
):
    """
    Prints a short window summary of AA (standard) vs AB (deviant) spike counts.
    """
    tones = np.asarray(tones)
    times = np.asarray(times)
    AA = np.where((tones[:-1] == 0) & (tones[1:] == 0))[0] + 1
    AB = np.where((tones[:-1] == 0) & (tones[1:] == 1))[0] + 1
    AA = AA[AA > exclude_first]
    AB = AB[AB > exclude_first]

    if len(AA) == 0 or len(AB) == 0:
        print("WARNING: Not enough AA/AB events after exclude_first.")
        return

    AA = AA[-n_trials:]
    AB = AB[-n_trials:]

    w0, w1 = (np.array(window_ms) * ms)
    if baseline_ms is None:
        b0 = b1 = None
    else:
        b0, b1 = (np.array(baseline_ms) * ms)

    def count_in_window(spmon, t_event, a0, a1):
        m = (spmon.t >= (t_event + a0)) & (spmon.t < (t_event + a1))
        return int(np.sum(m))

    def counts_for(spmon, idx_list):
        vals = []
        for idx in idx_list:
            t_ev = times[idx]
            t_ev = _as_ms_quantity(t_ev)
            c = count_in_window(spmon, t_ev, w0, w1)
            if b0 is not None:
                c -= count_in_window(spmon, t_ev, b0, b1)
            vals.append(c)
        return np.array(vals, dtype=float)

    key_for = {'pe': 'spikemon_pe', 'p': 'spikemon_p'}

    hdr = "=== MMN SUMMARY (Short Window) ==="
    sub = f"window=[{window_ms[0]}, {window_ms[1]}] ms | baseline=" + (
        "none" if baseline_ms is None else f"[{baseline_ms[0]}, {baseline_ms[1]}] ms")
    info = f"exclude_first={exclude_first}, AA_n={len(AA)}, AB_n={len(AB)}"
    print(hdr)
    print(sub)
    print(info)

    for lyr in layers:
        sm_key = key_for.get(lyr)
        smA = monitors_A.get(sm_key) if sm_key else None
        smB = monitors_B.get(sm_key) if sm_key else None
        if smA is None or smB is None:
            print(f"- {lyr.upper()}: '{sm_key}' not found, skipping.")
            continue

        AA_counts = counts_for(smA, AA)
        AB_counts = counts_for(smB, AB)

        mu_AA, sd_AA = float(np.mean(AA_counts)), float(np.std(AA_counts, ddof=1) if len(AA_counts) > 1 else 0.0)
        mu_AB, sd_AB = float(np.mean(AB_counts)), float(np.std(AB_counts, ddof=1) if len(AB_counts) > 1 else 0.0)
        
        d = np.nan
        if len(AA_counts) > 1 and len(AB_counts) > 1:
            pooled = np.sqrt(((len(AA_counts) - 1) * sd_AA ** 2 + (len(AB_counts) - 1) * sd_AB ** 2) / (
                    len(AA_counts) + len(AB_counts) - 2))
            d = (mu_AB - mu_AA) / pooled if pooled > 0 else np.nan

        print(f"\n[{lyr.upper()}]")
        print(f"AA mean±sd  : {mu_AA:.2f} ± {sd_AA:.2f}")
        print(f"AB mean±sd  : {mu_AB:.2f} ± {sd_AB:.2f}")
        print(f"Diff (AB-AA): {mu_AB - mu_AA:.2f}")
        print(f"Cohen's d   : {d:.2f}")

def _pick_late_events(tones, times, exclude_first=200):
    """
    Pick two events from late/stable period:
      - std_idx: 'AA' (standard preceded by standard)
      - dev_idx: 'AB' (deviant preceded by standard)
    """
    tones = np.asarray(tones)
    times = np.asarray(times)
    
    dev_indices = np.where((tones[:-1] == 0) & (tones[1:] == 1))[0] + 1
    dev_indices = dev_indices[dev_indices > exclude_first]
    if len(dev_indices) == 0:
        return None, None
    dev_idx = dev_indices[-1]

    std_candidates = np.where((tones[1:] == 0) & (tones[:-1] == 0))[0] + 1
    std_candidates = std_candidates[std_candidates > exclude_first]
    std_candidates = std_candidates[std_candidates < dev_idx]
    
    if len(std_candidates) == 0:
        return None, times[dev_idx]
    std_idx = std_candidates[-1]

    return times[std_idx], times[dev_idx]

def analyze_omission_response(mon_pe_A, tones, times, paradigm_params, window_start_ms=0, window_end_ms=250):
    """
    Quantifies Omission effect.
    """
    print("\n" + "-" * 50)
    print("--- OMISSION EFFECT QUANTITATIVE ANALYSIS ---")

    isi = paradigm_params['isi']
    start_window = window_start_ms * ms
    end_window = window_end_ms * ms

    t_standard_response_event = None
    t_omission_response_event = None

    for i in range(len(times) - 1):
        if abs((times[i + 1] - times[i]) - isi) < 0.01 * ms:
            t_standard_response_event = times[i + 1]
            break

    for i in range(len(times) - 1):
        if abs((times[i + 1] - times[i]) - (2 * isi)) < 0.01 * ms:
            t_omission_response_event = times[i] + isi
            break

    if t_standard_response_event is None or t_omission_response_event is None:
        print("WARNING: Could not find suitable 'AA' or 'A_' events for omission analysis.")
        print("-" * 50 + "\n")
        return

    std_mask = (mon_pe_A.t >= t_standard_response_event + start_window) & (
            mon_pe_A.t < t_standard_response_event + end_window)
    std_spike_count = len(mon_pe_A.t[std_mask])

    dev_mask = (mon_pe_A.t >= t_omission_response_event + start_window) & (
            mon_pe_A.t < t_omission_response_event + end_window)
    dev_spike_count = len(mon_pe_A.t[dev_mask])

    print(f"Analysis Window: {window_start_ms} ms to {window_end_ms} ms after event")
    print(f"Standard Response (2nd A in 'AA'): {std_spike_count} PE_A spikes")
    print(f"Omission Response ('A_' gap):      {dev_spike_count} PE_A spikes")

    if dev_spike_count > std_spike_count:
        print(">>> RESULT: Omission effect observed.")
    else:
        print(">>> RESULT: Expected omission effect NOT observed.")
    print("-" * 50 + "\n")

def _first_AB_pair(tones, times, t_min_ms=0, gap_target_ms=200, gap_tol_ms=10):
    tones = np.asarray(tones)
    idx = np.where((tones[:-1] == 0) & (tones[1:] == 1))[0]
    if len(idx) == 0:
        return None, None
    
    tA = np.array([float(times[i] / ms) for i in idx])
    tB = np.array([float(times[i + 1] / ms) for i in idx])
    m = (tA >= t_min_ms) & (np.abs((tB - tA) - gap_target_ms) <= gap_tol_ms)
    if not np.any(m):
        i = idx[0]
    else:
        i = idx[m][0]
    return times[i], times[i + 1]

def _count_spikes_in_window(spikemon, t0, start_ms=0, end_ms=150):
    if spikemon is None:
        return 0
    t_start = t0 + start_ms * ms
    t_end = t0 + end_ms * ms
    m = (spikemon.t >= t_start) & (spikemon.t < t_end)
    return int(np.sum(m))

def _psth(spikemon, t_ref, pre_ms=50, stim_ms=50, gap_ms=200, post_ms=50, bin_ms=2):
    if spikemon is None:
        return np.zeros(1), np.zeros(1)
    
    t0_abs = t_ref - pre_ms * ms
    t1_abs = t_ref + (pre_ms + stim_ms + gap_ms + stim_ms + post_ms) * ms
    m = (spikemon.t >= t0_abs) & (spikemon.t < t1_abs)
    t_rel_ms = (spikemon.t[m] - t_ref) / ms
    
    grid = np.arange(-pre_ms, pre_ms + stim_ms + gap_ms + stim_ms + post_ms + bin_ms, bin_ms, dtype=float)
    hist, _ = np.histogram(t_rel_ms, bins=grid)
    centers = 0.5 * (grid[:-1] + grid[1:])
    return centers, hist.astype(float)

def summarize_single_run(package):
    tones = package["tones"]
    times = package["times"]
    monA = package["monitors_A"]
    monB = package["monitors_B"]
    
    tA, tB = _first_AB_pair(tones, times, t_min_ms=0)
    if tA is None:
        return {
            "scalars": {"PE_A_0_150": np.nan, "PE_B_0_150": np.nan},
            "traces": {}
        }

    peA = monA.get("spikemon_pe")
    peB = monB.get("spikemon_pe")
    sA = _count_spikes_in_window(peA, tA, 0, 150)
    sB = _count_spikes_in_window(peB, tB, 0, 150)

    gridA, psthA = _psth(peA, tA, pre_ms=50, stim_ms=50, gap_ms=200, post_ms=50, bin_ms=2)
    gridB, psthB = _psth(peB, tA, pre_ms=50, stim_ms=50, gap_ms=200, post_ms=50, bin_ms=2)

    if gridA.shape != gridB.shape or not np.allclose(gridA, gridB):
        L = min(len(gridA), len(gridB))
        grid = gridA[:L]
        psthA = psthA[:L]
        psthB = psthB[:L]
    else:
        grid = gridA

    return {
        "scalars": {"PE_A_0_150": float(sA), "PE_B_0_150": float(sB)},
        "traces": {"psth_grid_ms": grid, "psth_PE_A": psthA, "psth_PE_B": psthB}
    }

def _combine_averages(summaries):
    keys_scalar = summaries[0]["scalars"].keys()
    avg_scalars = {k: float(np.nanmean([s["scalars"][k] for s in summaries])) for k in keys_scalar}

    grid = summaries[0]["traces"]["psth_grid_ms"]
    A_stack = np.vstack([s["traces"]["psth_PE_A"] for s in summaries])
    B_stack = np.vstack([s["traces"]["psth_PE_B"] for s in summaries])
    avg_traces = {
        "psth_grid_ms": grid,
        "psth_PE_A_mean": np.nanmean(A_stack, axis=0),
        "psth_PE_B_mean": np.nanmean(B_stack, axis=0),
        "psth_PE_A_std": np.nanstd(A_stack, axis=0),
        "psth_PE_B_std": np.nanstd(B_stack, axis=0),
    }
    return {"scalars": avg_scalars, "traces": avg_traces}

def mem_start_success_ratio(spikemon_mem_A_e,
                            A_times,
                            spikemon_mem_B_e,
                            B_times,
                            idxA0=0,
                            idxB0=0,
                            t_min=350000 * ms,
                            within=10 * ms):
    """
    Calculates success ratio of memory chain initiation.
    """
    
    def _filter_after(ts):
        return [t for t in ts if t >= t_min]

    def _count_starts(spmon, idx0, event_times):
        s_times = spmon.t[spmon.i == idx0]
        cnt = 0
        for t0 in event_times:
            if np.any((s_times >= t0) & (s_times < t0 + within)):
                cnt += 1
        return cnt, len(event_times)

    def _auto_detect_start_index(spmon, event_times):
        uniq = np.unique(spmon.i)
        if uniq.size == 0:
            return 0
        best_idx, best_hits = int(uniq[0]), -1
        for i in uniq:
            hits, _ = _count_starts(spmon, int(i), event_times)
            if hits > best_hits:
                best_idx, best_hits = int(i), hits
        return best_idx

    A_times_f = _filter_after(A_times)
    A_hits, A_total = _count_starts(spikemon_mem_A_e, idxA0, A_times_f)
    A_ratio = (A_hits / A_total) if A_total else 0.0
    print(f"[Mem Init|A] idx0={idxA0}  t>{int(t_min / ms)}ms  A={A_total}, "
          f"started={A_hits}, ratio={A_ratio:.2f}")

    result = {"A": {"idx0": idxA0, "total": A_total, "started": A_hits, "ratio": A_ratio}}

    if B_times is not None:
        if spikemon_mem_B_e is None:
            raise ValueError("spikemon_mem_B_e required for B measurement.")
        B_times_f = _filter_after(B_times)
        if idxB0 is None:
            idxB0 = _auto_detect_start_index(spikemon_mem_B_e, B_times_f)
            print(f"[Mem Init|B] idx0 auto-detected: {idxB0}")
        B_hits, B_total = _count_starts(spikemon_mem_B_e, idxB0, B_times_f)
        B_ratio = (B_hits / B_total) if B_total else 0.0
        print(f"[Mem Init|B] idx0={idxB0}  t>{int(t_min / ms)}ms  B={B_total}, "
              f"started={B_hits}, ratio={B_ratio:.2f}")
        result["B"] = {"idx0": idxB0, "total": B_total, "started": B_hits, "ratio": B_ratio}

    return result

def check_chain_triggers(spikemon_mem_e, tones, times, SOA, chain_delay,
                         module_name="A", window=None, slope_tol=0.5):
    """
    Checks if memory chain is triggered after each stimulus.
    """
    if window is None:
        window = 0.5 * SOA

    t_all = spikemon_mem_e.t
    i_all = spikemon_mem_e.i

    expected_slope = 1.0 / (chain_delay / ms)  # index / ms
    rows = []
    triggered = 0
    latencies = []

    for k, (tone, t0) in enumerate(zip(tones, times)):
        w0, w1 = t0, t0 + window
        m = (t_all >= w0) & (t_all < w1)
        if not np.any(m):
            rows.append({
                "stim_idx": k, "tone": int(tone), "t0_ms": float(t0 / ms),
                "triggered": False, "latency_ms": None, "start_i": None,
                "slope_idx_per_ms": None, "slope_ok": None
            })
            continue

        t_win = np.asarray(t_all[m] / ms, dtype=float)
        i_win = np.asarray(i_all[m], dtype=int)

        order = np.argsort(t_win)
        t_win = t_win[order]
        i_win = i_win[order]

        t_first = t_win[0]
        i_first = i_win[0]
        lat = float(t_first - (t0 / ms))

        # Zincir hızı
        if len(t_win) >= 3:
            x = t_win - t_first
            y = i_win - i_first
            try:
                slope = np.polyfit(x, y, 1)[0]
            except Exception:
                slope = (i_win[-1] - i_first) / max(1e-9, (t_win[-1] - t_first))
        elif len(t_win) >= 2:
            slope = (i_win[-1] - i_first) / max(1e-9, (t_win[-1] - t_first))
        else:
            slope = np.nan

        ok_slope = (abs(slope - expected_slope) <= slope_tol * expected_slope) if np.isfinite(slope) else None

        rows.append({
            "stim_idx": k, "tone": int(tone), "t0_ms": float(t0 / ms),
            "triggered": True, "latency_ms": lat, "start_i": int(i_first),
            "slope_idx_per_ms": float(slope) if np.isfinite(slope) else None,
            "slope_ok": bool(ok_slope) if ok_slope is not None else None
        })
        triggered += 1
        latencies.append(lat)

    n = len(times)
    summary = {
        "module": module_name,
        "N_stimuli": n,
        "triggered_count": triggered,
        "missed_count": n - triggered,
        "trigger_rate_%": 100.0 * triggered / n if n else 0.0,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else None,
        "median_latency_ms": float(np.median(latencies)) if latencies else None,
        "expected_slope_idx_per_ms": float(expected_slope)
    }
    return summary, rows

def report_missed(rows):
    """Reports missed triggers."""
    missed = [r for r in rows if not r.get("triggered", False)]
    if not missed:
        print("All stimuli triggered the chain.")
        return
    print(f"Missed: {len(missed)}")
    print("Indices:", [m['stim_idx'] for m in missed])
