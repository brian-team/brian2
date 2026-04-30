# PoC: Portable Network State Serialization for Brian2

**GSoC 2026 — Serialization/Deserialization for Brian Simulator models, results, and input data**

---

## What this does

This PoC implements the **core mechanism** of the project: serializing a
Brian2 Network state to a portable `.brian` ZIP archive and restoring it
with bit-identical numerical fidelity — without using Python pickle.

### The problem

`CPPStandaloneDevice.network_store()` and `network_restore()` both raise
`NotImplementedError`.  The existing `RuntimeDevice` checkpoint path uses
`pickle`, which is Python-version–specific, not human-readable, and
blocks interoperability with NWB/NEO/ML frameworks.

### The approach

Two new functions in `brian2/serialization/brian_format.py`:

| Function | What it does |
|---|---|
| `serialize_network_state(net, path)` | Calls `net._full_state()`, converts arrays + spike queues to a ZIP archive |
| `restore_network_state(net, path)` | Reads the archive and calls `obj._restore_from_full_state()` on each object |

**Archive layout** (`.brian` = ZIP):
```
checkpoint.brian
├── metadata.json    # format_version, brian2_version, t,
│                    # per-variable shapes / dtypes / SI dimension tuples
├── arrays.npz       # all ArrayVariable values (compressed NumPy binary)
└── spikequeues.json # SynapticPathway in-flight spike state
                     # (Cython SpikeQueue offset + spike_lists)
```

**Why this unlocks the full project:**

- For `CPPStandaloneDevice`: after a standalone run, arrays come from
  `results/<filename>` (read via `get_array_filename(var)`) instead of
  live NumPy arrays — the ZIP format is device-agnostic.
- For `BrianExporter` (brian2tools): `model.json` + `arrays.npz` become
  the interchange format for structural + state export.
- For `BrianImporter`: the JSON metadata is the reconstruction spec —
  object names, variable shapes, and SI dimension tuples (7-element list
  of SI exponents from `Dimension._dims`).

---

## Files

```
brian2/
└── serialization/
    ├── __init__.py          # exports serialize_network_state, restore_network_state
    └── brian_format.py      # core implementation (~170 lines)

examples/
└── serialization/
    └── brian_format_demo.py # end-to-end round-trip demo
```

---

## Run the demo

```bash
# from the repo root, with the brian2 dev environment active
conda activate brian2            # or: source .venv/bin/activate
python examples/serialization/brian_format_demo.py
```

Expected output:
```
Reference:  t=    10. ms  spikes=0  v[0,-1]=0.037158

Checkpoint: written to /tmp/....brian  (9.8 KB)
  Archive files : ['metadata.json', 'arrays.npz', 'spikequeues.json']
  Serialized t  : 5.0 ms
  Objects       : ['neurons', 'voltage', 'synapses_pre', 'spikes', 'synapses', 'defaultclock']

After restore: net.t=5. ms  G.v[0]=0.060653

Round-trip check:
  State variable max error : 0.00e+00
  Spike indices match      : True
  Spike times max error    : 0.00e+00

PASS — serialize → restore → run produces bit-identical results.
```

The demo:
1. Runs a LIF network (NeuronGroup + Synapses with per-neuron delay + StateMonitor + SpikeMonitor) for 5 ms.
2. Serializes to a `.brian` archive.
3. Resets to t=0 using the existing pickle store/restore.
4. Restores from the `.brian` archive to t=5 ms.
5. Runs 5 ms more and verifies the traces are **numerically identical** to a reference run that ran continuously to 10 ms.
