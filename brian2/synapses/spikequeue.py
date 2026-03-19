"""
SpikeQueue factory with automatic backend selection.

Priority: Cython (fastest, pre-compiled) > cppyy (JIT, no compiler needed)
"""

_backend = None

try:
    from .cythonspikequeue import SpikeQueue

    _backend = "cython"
except ImportError:
    try:
        from .cppyy_spikequeue import SpikeQueue

        _backend = "cppyy"
    except ImportError as e:
        raise ImportError(
            "No SpikeQueue backend available. Install either:\n"
            "  - Cython (recommended): pip install cython && pip install -e .\n"
            "  - cppyy (JIT fallback): pip install cppyy"
        ) from e

__all__ = ["SpikeQueue"]
