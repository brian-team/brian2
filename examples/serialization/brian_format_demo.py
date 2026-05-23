"""
Portable Network State Serialization — PoC Demo
================================================

This script demonstrates the core mechanism of the GSoC project:
serializing a Brian2 Network state to a .brian ZIP archive (JSON + npz)
and restoring it with exact numerical fidelity.

What it shows
-------------
1. A LIF network (NeuronGroup + Synapses + monitors) runs for 5 ms.
2. The network state is serialized to ``checkpoint.brian`` — a ZIP containing
   ``metadata.json`` (object/variable descriptors with SI dimensions) and
   ``arrays.npz`` (all raw array data, no pickle).
3. The network is reset to t=0 using the existing pickle-based store/restore.
4. The .brian checkpoint is loaded: all arrays and spike queues are restored
   to t=5 ms without re-running the simulation.
5. The network runs another 5 ms from the restored state.
6. The final traces are compared against a reference that ran continuously
   to 10 ms — they must be numerically identical.

Why this matters
----------------
- ``CPPStandaloneDevice.network_store/restore`` both raise NotImplementedError.
  The .brian format is the fix: after a C++ standalone run, arrays are read
  from ``results/<filename>`` (via ``get_array_filename``) and packed into
  the same ZIP layout — no code-gen changes required.
- Pickle is Python-version–specific. The .brian format is readable from any
  language and diffable in git.

Run
---
    cd /path/to/brian2
    python examples/serialization/brian_format_demo.py
"""

import json
import os
import tempfile
import zipfile

import numpy as np
from brian2 import *

from brian2.serialization import restore_network_state, serialize_network_state


TAU = 10 * ms  # module-level so it's in scope when net.run() resolves namespaces


def build_network(name_suffix=""):
    """Return a small deterministic LIF network ready to run."""
    G = NeuronGroup(
        20,
        "dv/dt = -v / TAU : 1",
        threshold="v > 0.9",
        reset="v = 0",
        method="exact",
        name=f"neurons{name_suffix}",
    )
    G.v = "0.1 + 0.05 * i / N"  # deterministic, no rand()

    S = Synapses(
        G, G,
        "w : 1",
        on_pre="v += w",
        name=f"synapses{name_suffix}",
    )
    S.connect(j="i")     # 1:1 deterministic connectivity
    S.w = 0.05
    S.delay = "i * 0.1 * ms"

    V = StateMonitor(G, "v", record=True, name=f"voltage{name_suffix}")
    M = SpikeMonitor(G, name=f"spikes{name_suffix}")
    return G, S, V, M


# ── Reference run (no checkpointing) ────────────────────────────────────────

start_scope()
G_ref, S_ref, V_ref, M_ref = build_network("_ref")
net_ref = Network(G_ref, S_ref, V_ref, M_ref)
net_ref.run(10 * ms)

ref_v       = V_ref.v[:, :].copy()
ref_spike_i = M_ref.i[:].copy()
ref_spike_t = M_ref.t_[:].copy()

print(f"Reference:  t={net_ref.t!s:>10s}  "
      f"spikes={len(ref_spike_i)}  "
      f"v[0,-1]={ref_v[0, -1]:.6f}")


# ── Checkpoint run (serialize at 5 ms, restore, continue) ───────────────────

start_scope()
G, S, V, M = build_network()
net = Network(G, S, V, M)

# Store initial state with the existing pickle mechanism so we can reset later
net.store("initial")

net.run(5 * ms)

# --- serialize to .brian archive -------------------------------------------
with tempfile.NamedTemporaryFile(suffix=".brian", delete=False) as f:
    ckpt_path = f.name

serialize_network_state(net, ckpt_path)

archive_size_kb = os.path.getsize(ckpt_path) / 1024
print(f"\nCheckpoint: written to {ckpt_path}  ({archive_size_kb:.1f} KB)")

# Verify archive structure
with zipfile.ZipFile(ckpt_path) as zf:
    names = zf.namelist()
    meta = json.loads(zf.read("metadata.json"))

print(f"  Archive files : {names}")
print(f"  Serialized t  : {meta['t'] * 1000:.1f} ms")
print(f"  Objects       : {list(meta['objects'].keys())}")

# --- reset to t=0, then restore from .brian archive ------------------------
net.restore("initial")           # existing pickle mechanism → t=0
restore_network_state(net, ckpt_path)  # .brian archive → t=5 ms

print(f"\nAfter restore: net.t={net.t!s}  G.v[0]={G.v[0]:.6f}")

# --- run from restored state -----------------------------------------------
net.run(5 * ms)

# ── Verify round-trip fidelity ───────────────────────────────────────────────

v_err       = np.max(np.abs(ref_v - V.v[:, :]))
spike_i_ok  = np.array_equal(ref_spike_i, M.i[:])
spike_t_err = (
    np.max(np.abs(ref_spike_t - M.t_[:])) if len(ref_spike_t) else 0.0
)

print(f"\nRound-trip check:")
print(f"  State variable max error : {v_err:.2e}")
print(f"  Spike indices match      : {spike_i_ok}")
print(f"  Spike times max error    : {spike_t_err:.2e}")

assert v_err < 1e-12,      f"State variable mismatch: {v_err}"
assert spike_i_ok,          "Spike index mismatch"
assert spike_t_err < 1e-12, f"Spike time mismatch: {spike_t_err}"

print("\nPASS — serialize → restore → run produces bit-identical results.")

os.unlink(ckpt_path)
