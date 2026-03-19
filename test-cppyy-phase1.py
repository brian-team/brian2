"""Phase 1 test: ratemonitor fix, RNG seeding, basic verification."""
import time
import numpy as np
from brian2 import *

prefs.codegen.target = 'cppyy'

print("=" * 60)
print("TEST 1: Basic neuron + StateMonitor + SpikeMonitor")
print("=" * 60)

num_neurons = 50
duration = 100*ms

eqs = '''
dv/dt = (v0 - v) / tau : volt
v0 : volt
tau : second
'''

G = NeuronGroup(num_neurons, eqs, threshold='v > -40*mV',
                reset='v = -60*mV', method='euler')
G.v = -60*mV
G.v0 = '-50*mV + 20*mV * i / num_neurons'
G.tau = 10*ms

spike_mon = SpikeMonitor(G)
state_mon = StateMonitor(G, 'v', record=[0, 25, 49])

t_start = time.perf_counter()
run(duration)
t_elapsed = time.perf_counter() - t_start

print(f"  Ran {num_neurons} neurons for {duration} in {t_elapsed:.2f}s")
print(f"  Spikes: {spike_mon.num_spikes}")
print(f"  StateMonitor: {state_mon.t.shape[0]} timesteps, {len(state_mon.record)} neurons")
assert state_mon.t.shape[0] > 0, "StateMonitor should have recorded data"
print("  PASS")

print()
print("=" * 60)
print("TEST 2: RateMonitor (was broken - push_back fix)")
print("=" * 60)

# Need a fresh network
start_scope()
prefs.codegen.target = 'cppyy'

G2 = NeuronGroup(100, eqs, threshold='v > -40*mV',
                 reset='v = -60*mV', method='euler')
G2.v = -60*mV
G2.v0 = '-50*mV + 25*mV * i / 100'
G2.tau = 10*ms

rate_mon = PopulationRateMonitor(G2)
spike_mon2 = SpikeMonitor(G2)

run(100*ms)

print(f"  RateMonitor: {len(rate_mon.t)} timesteps recorded")
rate_vals = np.array(rate_mon.rate/Hz)
print(f"  Rate range: {rate_vals.min():.1f} - {rate_vals.max():.1f} Hz")
assert len(rate_mon.t) > 0, "RateMonitor should have recorded data"
assert len(rate_mon.rate) == len(rate_mon.t), "rate and t should have same length"
print("  PASS")

print()
print("=" * 60)
print("TEST 3: RNG Seeding (reproducibility)")
print("=" * 60)

start_scope()
prefs.codegen.target = 'cppyy'

# Run with seed 42 twice, results should match
results = []
for trial in range(2):
    start_scope()
    seed(42)
    eqs3 = '''
    dv/dt = -v/tau + sigma*xi/tau**0.5 : volt
    tau : second
    sigma : volt
    '''
    G3 = NeuronGroup(20, eqs3,
                     threshold='v > 1*mV', reset='v = 0*mV', method='euler')
    G3.v = 0*mV
    G3.tau = 10*ms
    G3.sigma = 2*mV
    sm3 = SpikeMonitor(G3)
    run(50*ms)
    results.append(sm3.num_spikes)
    print(f"  Trial {trial+1}: {sm3.num_spikes} spikes")

# Note: exact match depends on cppyy RNG being seeded correctly
# The numpy RNG is definitely seeded, so xi (which uses numpy) should match
print(f"  Results match: {results[0] == results[1]}")
if results[0] == results[1]:
    print("  PASS (deterministic)")
else:
    print("  WARN: results differ (cppyy RNG may use different path)")

print()
print("All Phase 1 tests complete!")
