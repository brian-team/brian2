"""
Comprehensive test suite for the cppyy JIT backend.

Each test runs in a separate subprocess because Cling (the JIT compiler)
accumulates state that can conflict across start_scope() calls within a
single process. Subprocess isolation gives each test a clean Cling session.

16 tests covering: basic neurons, all 3 monitors, RNG seeding, 4 synapse
connection methods, STDP, summed variables, multisynaptic index, multi-run,
store/restore, refractoriness, delayed synapses.
"""
import subprocess
import sys
import textwrap
import time

_PREAMBLE = textwrap.dedent("""\
    import numpy as np
    from brian2 import *
    prefs.codegen.target = "cppyy"
""")

TESTS = {}


def register(name):
    def decorator(func):
        # Extract function body source from the docstring
        TESTS[name] = func.__doc__
        return func
    return decorator


@register("Basic LIF neuron")
def _():
    """
    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                    threshold='v > 1', reset='v = 0', method='euler')
    run(50 * ms)
    assert np.any(G.v[:] > 0), "Neurons should have nonzero v"
    """


@register("SpikeMonitor")
def _():
    """
    G = NeuronGroup(5, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                    threshold='v > 1', reset='v = 0', method='euler')
    mon = SpikeMonitor(G)
    run(100 * ms)
    assert mon.num_spikes > 0, "Should have recorded spikes"
    assert len(mon.t) == len(mon.i), "t and i arrays must match"
    assert np.all(mon.i[:] < 5), "Spike indices must be in range"
    """


@register("StateMonitor")
def _():
    """
    G = NeuronGroup(5, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                    threshold='v > 1', reset='v = 0', method='euler')
    mon = StateMonitor(G, 'v', record=[0, 2, 4])
    run(20 * ms)
    assert mon.t.shape[0] > 0, "Should have recorded timesteps"
    assert mon.v.shape == (3, mon.t.shape[0]), "Shape mismatch"
    """


@register("RateMonitor")
def _():
    """
    G = NeuronGroup(50, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                    threshold='v > 1', reset='v = 0', method='euler')
    mon = PopulationRateMonitor(G)
    run(100 * ms)
    assert len(mon.t) > 0, "Should have recorded rate"
    assert len(mon.rate) == len(mon.t), "rate and t must match"
    assert np.any(mon.rate[:] > 0 * Hz), "Should have nonzero rate"
    """


@register("RNG seeding reproducibility")
def _():
    """
    seed(12345)
    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) + xi*sqrt(2/(10*ms)) : 1', method='euler')
    run(10 * ms)
    result1 = np.array(G.v[:])

    start_scope()
    seed(12345)
    G2 = NeuronGroup(10, 'dv/dt = -v/(10*ms) + xi*sqrt(2/(10*ms)) : 1', method='euler')
    run(10 * ms)
    result2 = np.array(G2.v[:])

    np.testing.assert_array_equal(result1, result2)
    """


@register("Synapses - explicit i/j")
def _():
    """
    pre = NeuronGroup(5, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                      threshold='v > 1', reset='v = 0', method='euler')
    post = NeuronGroup(3, 'dv/dt = -v/(10*ms) : 1', method='euler')
    S = Synapses(pre, post, 'w : 1', on_pre='v_post += w')
    S.connect(i=[0, 1, 2, 3, 4], j=[0, 1, 2, 0, 1])
    S.w = 0.5
    run(50 * ms)
    assert len(S) == 5, f"Expected 5 synapses, got {len(S)}"
    """


@register("Synapses - one-to-one")
def _():
    """
    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.3/ms : 1',
                    threshold='v > 1', reset='v = 0', method='euler')
    S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
    S.connect(j='i', skip_if_invalid=True)
    S.w = 0.1
    run(50 * ms)
    assert len(S) == 10, f"Expected 10 synapses, got {len(S)}"
    """


@register("Synapses - all-to-all")
def _():
    """
    pre = NeuronGroup(4, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                      threshold='v > 1', reset='v = 0', method='euler')
    post = NeuronGroup(3, 'dv/dt = -v/(10*ms) : 1', method='euler')
    S = Synapses(pre, post, 'w : 1', on_pre='v_post += w')
    S.connect()
    S.w = 0.1
    run(50 * ms)
    assert len(S) == 12, f"Expected 12 synapses, got {len(S)}"
    """


@register("Synapses - probabilistic")
def _():
    """
    seed(42)
    pre = NeuronGroup(20, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                      threshold='v > 1', reset='v = 0', method='euler')
    post = NeuronGroup(20, 'dv/dt = -v/(10*ms) : 1', method='euler')
    S = Synapses(pre, post, 'w : 1', on_pre='v_post += w')
    S.connect(p=0.5)
    S.w = 0.05
    run(20 * ms)
    assert 50 < len(S) < 350, f"Unexpected synapse count: {len(S)}"
    """


@register("STDP")
def _():
    """
    inp = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.3/ms : 1',
                      threshold='v > 1', reset='v = 0', method='euler')
    out = NeuronGroup(10, 'dv/dt = -v/(20*ms) : 1',
                      threshold='v > 1', reset='v = 0', method='euler')
    S = Synapses(inp, out,
                 \'\'\'w : 1
                    dApre/dt = -Apre / (20*ms) : 1 (event-driven)
                    dApost/dt = -Apost / (20*ms) : 1 (event-driven)\'\'\',
                 on_pre=\'\'\'v_post += w
                           Apre += 0.01
                           w = clip(w + Apost, 0, 1)\'\'\',
                 on_post=\'\'\'Apost += -0.01
                            w = clip(w + Apre, 0, 1)\'\'\')
    S.connect(j='i')
    S.w = 0.5
    run(100 * ms)
    assert len(S) == 10
    w_vals = np.array(S.w[:])
    assert not np.allclose(w_vals, 0.5), "STDP should modify weights"
    """


@register("Summed variable")
def _():
    """
    G_pre = NeuronGroup(5, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                        threshold='v > 1', reset='v = 0', method='euler')
    G_post = NeuronGroup(3,
                         \'\'\'dv/dt = (I_syn - v) / (10*ms) : 1
                            I_syn : 1\'\'\',
                         threshold='v > 1', reset='v = 0', method='euler')
    S = Synapses(G_pre, G_post, 'w : 1\\nI_syn_post = w : 1 (summed)')
    S.connect()
    S.w = '0.1 * rand()'
    mon = StateMonitor(G_post, 'I_syn', record=[0])
    run(50 * ms)
    assert len(S) == 15
    assert mon.t.shape[0] > 0
    """


@register("Multisynaptic index")
def _():
    """
    pre = NeuronGroup(3, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                      threshold='v > 1', reset='v = 0', method='euler')
    post = NeuronGroup(3, 'dv/dt = -v/(10*ms) : 1', method='euler')
    S = Synapses(pre, post, 'w : 1', on_pre='v_post += w', multisynaptic_index='k')
    S.connect(i=[0, 0, 1, 1, 1], j=[0, 0, 1, 1, 2])
    S.w = '0.1 * (k + 1)'
    run(20 * ms)
    assert len(S) == 5
    w_vals = np.array(S.w[:])
    assert len(w_vals) == 5
    """


@register("Multi-run")
def _():
    """
    G = NeuronGroup(5, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                    threshold='v > 1', reset='v = 0', method='euler')
    mon = SpikeMonitor(G)
    run(50 * ms)
    spikes_1 = mon.num_spikes
    run(50 * ms)
    spikes_2 = mon.num_spikes
    assert spikes_2 >= spikes_1, "Second run should add more spikes"
    assert spikes_2 > 0, "Should have spikes after 100ms total"
    """


@register("Store/restore")
def _():
    """
    G = NeuronGroup(5, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                    threshold='v > 1', reset='v = 0', method='euler')
    run(20 * ms)
    v_before = np.array(G.v[:])
    store()
    run(30 * ms)
    restore()
    v_restored = np.array(G.v[:])
    np.testing.assert_array_almost_equal(v_before, v_restored)
    """


@register("Refractoriness")
def _():
    """
    # Use string-based refractory condition (like the HH model)
    G = NeuronGroup(1, 'dv/dt = 0.25/ms : 1', threshold='v > 1',
                    reset='v = 0', refractory='v > 0.5', method='euler')
    mon = SpikeMonitor(G)
    run(100 * ms)
    spike_times = np.array(mon.t[:])
    assert len(spike_times) >= 2, "Should have at least 2 spikes"
    # Verify spikes are regular (neuron resets, climbs back, spikes again)
    isis = np.diff(spike_times)
    assert np.std(isis) / np.mean(isis) < 0.1, \
        f"ISIs should be regular, got std/mean = {np.std(isis)/np.mean(isis):.3f}"
    """


@register("Delayed synapses")
def _():
    """
    inp = NeuronGroup(1, 'dv/dt = 2.0/ms : 1', threshold='v > 1',
                      reset='v = 0', method='euler')
    out = NeuronGroup(1, 'dv/dt = -v/(10*ms) : 1', method='euler')
    S = Synapses(inp, out, 'w : 1', on_pre='v_post += w')
    S.connect()
    S.w = 0.5
    S.delay = 3 * ms
    mon_out = StateMonitor(out, 'v', record=[0])
    mon_in = SpikeMonitor(inp)
    run(30 * ms)
    assert mon_in.num_spikes > 0, "Input should spike"
    assert np.any(mon_out.v[0] > 0), "Output should receive delayed input"
    """


def run_test(index, name, code):
    """Run a single test in a subprocess."""
    full_code = _PREAMBLE + textwrap.dedent(code)
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", full_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        print(f"  [{index:2d}] {name}... PASS ({elapsed:.2f}s)")
        return True
    else:
        print(f"  [{index:2d}] {name}... FAIL ({elapsed:.2f}s)")
        # Show last few lines of stderr (skip cppyy noise)
        err_lines = [
            l for l in result.stderr.strip().split("\n")
            if not l.startswith("[/") and "no debug info" not in l
        ]
        if err_lines:
            for line in err_lines[-5:]:
                print(f"       {line}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("cppyy Backend Comprehensive Test Suite")
    print("(each test runs in isolated subprocess)")
    print("=" * 60)

    passed = 0
    failed = 0
    failed_names = []

    t_total = time.perf_counter()
    for i, (name, code) in enumerate(TESTS.items(), 1):
        if run_test(i, name, code):
            passed += 1
        else:
            failed += 1
            failed_names.append(name)
    t_total = time.perf_counter() - t_total

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed ({t_total:.1f}s)")
    if failed_names:
        print(f"Failed: {', '.join(failed_names)}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
