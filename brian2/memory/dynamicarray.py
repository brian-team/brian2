"""
DynamicArray factory with automatic backend selection.

Priority: Cython (fastest, pre-compiled) > cppyy (JIT, no compiler needed)

Both backends expose the same API: .data, .resize(), .shrink(),
.get_capsule(), __getitem__, __setitem__, __len__, .shape.
"""

_backend = None

try:
    from .cythondynamicarray import DynamicArray, DynamicArray1D

    _backend = "cython"
except ImportError:
    try:
        from .cppyy_dynamicarray import DynamicArray, DynamicArray1D

        _backend = "cppyy"
    except ImportError as e:
        raise ImportError(
            "No DynamicArray backend available. Install either:\n"
            "  - Cython (recommended): pip install cython && pip install -e .\n"
            "  - cppyy (JIT fallback): pip install cppyy"
        ) from e

__all__ = ["DynamicArray", "DynamicArray1D", "_backend"]
