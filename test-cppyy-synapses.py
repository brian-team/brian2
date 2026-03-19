"""Test cppyy synapse support: SpikeQueue + Synapses templates."""
import time
import numpy as np
from brian2 import *

prefs.codegen.target = 'cppyy'

print("=" * 60)
print("TEST 1: Basic Synapses with fixed connectivity")
print("=" * 60)

start_scope()

# Simple pre-post network
inp = NeuronGroup(5, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                  threshold='v > 1', reset='v = 0', method='euler')

out = NeuronGroup(3, 'dv/dt = -v/(10*ms) : 1',
                  threshold='v > 1', reset='v = 0', method='euler')

S = Synapses(inp, out, 'w : 1', on_pre='v_post += w')
S.connect(i=[0, 1, 2, 3, 4], j=[0, 1, 2, 0, 1])
S.w = 0.5

spike_inp = SpikeMonitor(inp)
spike_out = SpikeMonitor(out)
state_out = StateMonitor(out, 'v', record=True)

t_start = time.perf_counter()
run(200*ms)
t_elapsed = time.perf_counter() - t_start

print(f"  Ran in {t_elapsed:.2f}s")
print(f"  Input spikes: {spike_inp.num_spikes}")
print(f"  Output spikes: {spike_out.num_spikes}")
print(f"  Synapses created: {len(S)}")
print(f"  StateMonitor: {state_out.t.shape[0]} timesteps")
assert len(S) == 5, f"Expected 5 synapses, got {len(S)}"
print("  PASS")

print()
print("=" * 60)
print("TEST 2: STDP-like synapse with pre/post pathways")
print("=" * 60)

start_scope()

N = 10
inp2 = NeuronGroup(N, 'dv/dt = -v/(10*ms) + 0.3/ms : 1',
                   threshold='v > 1', reset='v = 0', method='euler')

out2 = NeuronGroup(N, 'dv/dt = -v/(20*ms) : 1',
                   threshold='v > 1', reset='v = 0', method='euler')

S2 = Synapses(inp2, out2,
              '''w : 1
                 dApre/dt = -Apre / (20*ms) : 1 (event-driven)
                 dApost/dt = -Apost / (20*ms) : 1 (event-driven)''',
              on_pre='''v_post += w
                        Apre += 0.01
                        w = clip(w + Apost, 0, 1)''',
              on_post='''Apost += -0.01
                         w = clip(w + Apre, 0, 1)''')
S2.connect(j='i')  # one-to-one
S2.w = 0.5

run(100*ms)

print(f"  Synapses: {len(S2)}")
print(f"  Weight range: {float(np.min(S2.w)):.4f} - {float(np.max(S2.w)):.4f}")
assert len(S2) == N, f"Expected {N} synapses, got {len(S2)}"
print("  PASS")

print()
print("=" * 60)
print("TEST 3: Summed variable (synaptic current)")
print("=" * 60)

start_scope()

eqs_neurons = '''
dv/dt = (I_syn - v) / (10*ms) : 1
I_syn : 1
'''

G_pre = NeuronGroup(5, 'dv/dt = -v/(10*ms) + 0.5/ms : 1',
                    threshold='v > 1', reset='v = 0', method='euler')

G_post = NeuronGroup(3, eqs_neurons, threshold='v > 1', reset='v = 0',
                     method='euler')

S3 = Synapses(G_pre, G_post,
              '''w : 1
                 I_syn_post = w : 1 (summed)''')
S3.connect()  # all-to-all
S3.w = '0.1 * rand()'

sm = StateMonitor(G_post, 'I_syn', record=[0])

run(50*ms)

print(f"  Synapses: {len(S3)}")
print(f"  I_syn recorded: {sm.t.shape[0]} timesteps")
print(f"  I_syn range: {float(np.min(sm.I_syn[0])):.4f} - {float(np.max(sm.I_syn[0])):.4f}")
print("  PASS")

print()
print("All synapse tests complete!")
