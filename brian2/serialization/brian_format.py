"""
brian2.serialization.brian_format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Serialize and restore Brian2 Network state as a portable ``.brian`` ZIP
archive (JSON metadata + NumPy ``.npz`` arrays) instead of Python pickle.

This module demonstrates the core mechanism for:

1. **Fixing** ``CPPStandaloneDevice.network_store / network_restore``
   (currently raise ``NotImplementedError``): after a standalone run, the
   same ZIP format is populated by reading binary files from ``results/``
   via ``get_array_filename(var)`` rather than from live NumPy arrays.

2. **Replacing** pickle in ``RuntimeDevice`` store/restore with an
   interoperable format that is not tied to a Python version or platform.

Archive layout
--------------
::

    checkpoint.brian   (ZIP, deflate)
    ├── metadata.json      # format_version, brian2_version, t, per-object
    │                      # variable shapes / dtypes / SI dimension tuples
    ├── arrays.npz         # flat dict  "ObjName__varname" → ndarray
    └── spikequeues.json   # SynapticPathway _spikequeue states
                           # (offset + spike_lists, JSON-serialisable)

Why not pickle?
---------------
Pickle is Python-version–specific, not human-readable, and blocks
interoperability with tools such as NWB, NEO, or ML frameworks.  A ZIP of
JSON + npz is readable in any language, diff-able in git, and extensible
without breaking existing archives.
"""

import io
import json
import zipfile

import numpy as np

import brian2

_METADATA_FILE = "metadata.json"
_ARRAYS_FILE = "arrays.npz"
_QUEUES_FILE = "spikequeues.json"
_FORMAT_VERSION = "1"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def serialize_network_state(net, filepath):
    """
    Serialize the state of *net* to a ``.brian`` ZIP archive at *filepath*.

    Parameters
    ----------
    net : brian2.Network
        The network whose state to serialize.  Must have been run (or at
        least initialized) so that all ``ArrayVariable`` objects hold
        concrete values.
    filepath : str or path-like
        Destination path, e.g. ``"checkpoint.brian"``.

    Notes
    -----
    Internally this calls ``net._full_state()``, the same method used by
    the existing ``Network.store()`` / ``Network.restore()`` pickle path,
    so fidelity guarantees are identical.

    For ``CPPStandaloneDevice`` the same JSON + npz layout is used, but
    arrays are read from ``results/<filename>`` via
    ``CPPStandaloneDevice.get_array_filename(var)`` rather than from live
    NumPy arrays — the format is device-agnostic by design.
    """
    # net._full_state() walks all objects via _get_all_objects() and calls
    # obj._full_state() on each VariableOwner, plus every unique Clock.
    # Returns: {obj_name: {var_name: (values_copy, size)}, "0_t": float}
    raw_state = net._full_state()

    t = float(raw_state.pop("0_t"))

    metadata = {
        "format_version": _FORMAT_VERSION,
        "brian2_version": brian2.__version__,
        "t": t,
        "objects": {},
        "spikequeue_objects": [],
    }

    arrays = {}       # flat key "ObjName__varname" → ndarray
    spikequeues = {}  # obj_name → encoded queue state

    for obj_name, obj_state in raw_state.items():
        obj_meta = {}

        for var_name, value in obj_state.items():
            if var_name == "_spikequeue":
                # SynapticPathway stores queue as (offset, list_of_lists)
                # or None when the pathway has no in-flight spikes yet.
                encoded = _encode_spikequeue(value)
                if encoded is not None:
                    spikequeues[obj_name] = encoded
                    metadata["spikequeue_objects"].append(obj_name)
                continue

            arr, size = value
            arr = np.asarray(arr)
            key = f"{obj_name}__{var_name}"
            arrays[key] = arr
            # size is int for 1-D arrays and a tuple for 2-D arrays
            # (e.g. StateMonitor records shape (timesteps, N_neurons)).
            size_json = list(size) if isinstance(size, tuple) else int(size)
            obj_meta[var_name] = {
                "size": size_json,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                # Preserve SI dimension as a 7-tuple of exponents so the
                # archive can be read without Brian2 installed (e.g. in
                # analysis scripts or by NWB/NEO converters).
                "dim": _dim_of_array_key(net, obj_name, var_name),
            }

        metadata["objects"][obj_name] = obj_meta

    # Pack into ZIP: JSON text + compressed npz blob
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, **arrays)

    with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_METADATA_FILE, json.dumps(metadata, indent=2))
        zf.writestr(_ARRAYS_FILE, npz_buf.getvalue())
        zf.writestr(_QUEUES_FILE, json.dumps(spikequeues, indent=2))


def restore_network_state(net, filepath):
    """
    Restore the state of *net* from a ``.brian`` ZIP archive at *filepath*.

    The network's objects must already exist and carry the **same names**
    as when the archive was created — the same contract as the existing
    ``Network.store() / Network.restore()`` mechanism.

    Parameters
    ----------
    net : brian2.Network
    filepath : str or path-like
    """
    with zipfile.ZipFile(filepath, "r") as zf:
        metadata = json.loads(zf.read(_METADATA_FILE))
        npz_data = np.load(
            io.BytesIO(zf.read(_ARRAYS_FILE)), allow_pickle=False
        )
        spikequeues = json.loads(zf.read(_QUEUES_FILE))

    # Restore the network simulation time directly (mirrors Network.restore)
    net.t_ = float(metadata["t"])

    # Rebuild the state dict that _restore_from_full_state expects:
    # {var_name: (ndarray, size)}  plus "_spikequeue" for pathways
    reconstructed = {}
    for obj_name, obj_meta in metadata["objects"].items():
        obj_state = {}
        for var_name, var_info in obj_meta.items():
            key = f"{obj_name}__{var_name}"
            raw_size = var_info["size"]
            # Restore original type: list → tuple, int stays int
            size = tuple(raw_size) if isinstance(raw_size, list) else raw_size
            obj_state[var_name] = (npz_data[key], size)
        if obj_name in spikequeues:
            obj_state["_spikequeue"] = _decode_spikequeue(spikequeues[obj_name])
        reconstructed[obj_name] = obj_state

    # Walk all live network objects (same traversal as Network._full_state)
    from brian2.core.network import _get_all_objects

    all_objects = _get_all_objects(net.objects)
    clocks = {obj.clock for obj in all_objects}

    for obj in list(all_objects) + list(clocks):
        if obj.name in reconstructed:
            obj._restore_from_full_state(reconstructed[obj.name])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_spikequeue(queue_state):
    """
    Convert a Cython ``SpikeQueue._full_state()`` return value to a
    JSON-serializable dict.

    Cython format: ``(offset: int, spike_lists: list[list[int]])``
    """
    if queue_state is None:
        return None
    offset, spike_lists = queue_state
    return {
        "offset": int(offset),
        "spike_lists": [[int(s) for s in slot] for slot in spike_lists],
    }


def _decode_spikequeue(encoded):
    """
    Reconstruct the Cython ``SpikeQueue`` 2-tuple from a JSON-loaded dict.

    Compatible with ``SynapticPathway._restore_from_full_state`` which
    calls ``self.queue._restore_from_full_state(converted_queue_state)``.
    """
    if encoded is None:
        return None
    return (encoded["offset"], [list(slot) for slot in encoded["spike_lists"]])


def _dim_of_array_key(net, obj_name, var_name):
    """
    Look up the SI dimension of *var_name* on the object named *obj_name*
    and return it as a plain Python list (7 SI exponents).

    Falls back to ``None`` if the variable or its dimension cannot be
    resolved — this is metadata only and does not affect restore fidelity.
    """
    from brian2.core.network import _get_all_objects
    from brian2.core.clocks import Clock

    all_objects = list(_get_all_objects(net.objects))
    clocks = list({obj.clock for obj in all_objects})
    for obj in all_objects + clocks:
        if obj.name != obj_name:
            continue
        if not hasattr(obj, "variables"):
            return None
        var = obj.variables.get(var_name)
        if var is None:
            return None
        dim = getattr(var, "dim", None)
        if dim is None:
            return None
        try:
            return list(dim._dims)
        except AttributeError:
            return None
    return None
