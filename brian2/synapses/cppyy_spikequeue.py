"""
cppyy-backed SpikeQueue implementation.

Drop-in replacement for the Cython SpikeQueue wrapper. Uses cppyy to
JIT-compile and instantiate CSpikeQueue from spikequeue.h directly.

API matches the Cython version: prepare(), push(), peek(), advance(),
get_capsule(), _full_state(), _restore_from_full_state().
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

import numpy as np

_cppyy = None
_initialized = False

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def _ensure_initialized():
    global _cppyy, _initialized
    if _initialized:
        return

    import cppyy

    _cppyy = cppyy

    import brian2

    # spikequeue.h is in the synapses directory
    synapses_path = os.path.join(os.path.dirname(brian2.__file__), "synapses")
    cppyy.add_include_path(synapses_path)

    # stdint_compat.h is in brianlib
    brianlib_path = os.path.join(
        os.path.dirname(brian2.__file__), "devices", "cpp_standalone", "brianlib"
    )
    cppyy.add_include_path(brianlib_path)

    cppyy.include("spikequeue.h")

    # Helper to get peek() result as raw pointer + size
    cppyy.cppdef(
        """
    #ifndef _BRIAN2_CPPYY_SPIKEQUEUE_HELPERS
    #define _BRIAN2_CPPYY_SPIKEQUEUE_HELPERS
    #include <cstdint>

    uintptr_t _brian_sq_peek_data(CSpikeQueue& sq, size_t& out_size) {
        std::vector<int32_t>* v = sq.peek();
        out_size = v->size();
        if (out_size == 0) return 0;
        return reinterpret_cast<uintptr_t>(&(*v)[0]);
    }

    void _brian_sq_push_array(CSpikeQueue& sq, uintptr_t data_addr, int count) {
        sq.push(reinterpret_cast<int32_t*>(data_addr), count);
    }

    template<typename scalar>
    void _brian_sq_prepare(CSpikeQueue& sq, uintptr_t delays_addr, int n_delays,
                           uintptr_t sources_addr, int n_synapses, double dt) {
        sq.prepare<scalar>(
            reinterpret_cast<scalar*>(delays_addr), n_delays,
            reinterpret_cast<int32_t*>(sources_addr), n_synapses, dt);
    }
    #endif
    """
    )

    _initialized = True


class SpikeQueue:
    """cppyy-backed SpikeQueue. API-compatible with Cython SpikeQueue."""

    def __init__(self, source_start: int, source_end: int):
        _ensure_initialized()
        self._cpp_obj = _cppyy.gbl.CSpikeQueue(int(source_start), int(source_end))
        self._state_tuple = (source_start, source_end, np.int32)

    def prepare(self, real_delays: np.ndarray, dt: float, sources: np.ndarray):
        real_delays = np.ascontiguousarray(real_delays)
        sources = np.ascontiguousarray(sources, dtype=np.int32)

        delays_addr = real_delays.ctypes.data
        sources_addr = sources.ctypes.data

        if real_delays.dtype == np.float32:
            _cppyy.gbl._brian_sq_prepare["float"](
                self._cpp_obj,
                delays_addr,
                len(real_delays),
                sources_addr,
                len(sources),
                float(dt),
            )
        else:
            _cppyy.gbl._brian_sq_prepare["double"](
                self._cpp_obj,
                delays_addr,
                len(real_delays),
                sources_addr,
                len(sources),
                float(dt),
            )

    def push(self, spikes: np.ndarray):
        spikes = np.ascontiguousarray(spikes, dtype=np.int32)
        _cppyy.gbl._brian_sq_push_array(self._cpp_obj, spikes.ctypes.data, len(spikes))

    def peek(self) -> np.ndarray:
        size = _cppyy.gbl.size_t(0)
        addr = int(_cppyy.gbl._brian_sq_peek_data(self._cpp_obj, size))
        n = int(size)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        # Zero-copy view into the vector's data
        ctype = ctypes.c_int32 * n
        c_arr = ctype.from_address(addr)
        return np.ctypeslib.as_array(c_arr).copy()  # copy — vector may reallocate

    def advance(self):
        self._cpp_obj.advance()

    def get_capsule(self) -> Any:
        addr = _cppyy.addressof(self._cpp_obj)
        return _PyCapsule_New(addr, b"CSpikeQueue", None)

    def _full_state(self):
        return self._cpp_obj._full_state()

    def _restore_from_full_state(self, state):
        if state is not None:
            self._cpp_obj._restore_from_full_state(state)
        else:
            import cppyy

            empty_queue = cppyy.gbl.std.vector["std::vector<int32_t>"]()
            empty_state = cppyy.gbl.std.pair["int, std::vector<std::vector<int32_t>>"](
                0, empty_queue
            )
            self._cpp_obj._restore_from_full_state(empty_state)
