"""
cppyy-backed DynamicArray implementations.

Drop-in replacement for the Cython wrappers in cythondynamicarray.pyx.
Uses cppyy to JIT-compile and instantiate the same C++ DynamicArray1D<T>
and DynamicArray2D<T> classes from brianlib/dynamic_array.h.

The API matches the Cython version exactly: .data, .resize(), .shrink(),
.get_capsule(), __getitem__, __setitem__, __len__, .shape.

PyCapsule names are identical ("DynamicArray1D" / "DynamicArray2D") so
cppyy templates that extract pointers from capsules work unchanged.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import Any

import numpy as np

# Lazy init — cppyy is only imported when this module is actually used
_cppyy = None
_initialized = False

# PyCapsule_New from Python C API — used to create capsules without Cython
_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

# Map numpy dtypes to C++ type names for template instantiation
_DTYPE_TO_CPP = {
    np.dtype(np.float64): "double",
    np.dtype(np.float32): "float",
    np.dtype(np.int32): "int32_t",
    np.dtype(np.int64): "int64_t",
    np.dtype(np.bool_): "char",  # C++ bool packs bits; use char for byte-level access
}


def _ensure_initialized():
    """Load cppyy and include dynamic_array.h exactly once."""
    global _cppyy, _initialized
    if _initialized:
        return

    import cppyy

    _cppyy = cppyy

    import brian2

    brianlib_path = os.path.join(
        os.path.dirname(brian2.__file__), "devices", "cpp_standalone", "brianlib"
    )
    cppyy.add_include_path(brianlib_path)
    cppyy.include("dynamic_array.h")

    # Helper to get raw data address as uintptr_t (works for all types including char*)
    cppyy.cppdef(
        """
    #ifndef _BRIAN2_CPPYY_DYNARRAY_HELPERS
    #define _BRIAN2_CPPYY_DYNARRAY_HELPERS
    template<typename T>
    uintptr_t _brian_dynarray_data_addr_1d(DynamicArray1D<T>& arr) {
        return reinterpret_cast<uintptr_t>(arr.get_data_ptr());
    }
    template<typename T>
    uintptr_t _brian_dynarray_data_addr_2d(DynamicArray2D<T>& arr) {
        return reinterpret_cast<uintptr_t>(arr.get_data_ptr());
    }
    #endif
    """
    )

    _initialized = True


class CppyyDynamicArray1D:
    """
    cppyy-backed 1D dynamic array. API-compatible with Cython DynamicArray1DClass.
    """

    def __init__(self, initial_size: int, dtype=np.float64, factor: float = 2.0):
        _ensure_initialized()
        self.dtype = np.dtype(dtype)
        self._factor = factor

        cpp_type = _DTYPE_TO_CPP.get(self.dtype)
        if cpp_type is None:
            raise TypeError(f"Unsupported dtype: {self.dtype}")

        # Instantiate DynamicArray1D<T>(size, factor) via cppyy
        cls = getattr(_cppyy.gbl, f"DynamicArray1D<{cpp_type}>")
        self._cpp_obj = cls(int(initial_size), float(factor))
        self._cpp_type = cpp_type

    def resize(self, new_size: int) -> None:
        self._cpp_obj.resize(int(new_size))

    def shrink(self, new_size: int) -> None:
        self._cpp_obj.shrink(int(new_size))

    def get_capsule(self) -> Any:
        """Return PyCapsule wrapping the C++ DynamicArray1D<T>* pointer."""
        # Get the raw address of the C++ object via cppyy
        addr = _cppyy.addressof(self._cpp_obj)
        return _PyCapsule_New(addr, b"DynamicArray1D", None)

    @property
    def data(self) -> np.ndarray:
        """Zero-copy numpy view of the underlying buffer."""
        size = self._cpp_obj.size()
        if size == 0:
            return np.array([], dtype=self.dtype)

        # Use C++ helper to get raw address (works for all types including char*)
        addr = int(
            _cppyy.gbl._brian_dynarray_data_addr_1d[self._cpp_type](self._cpp_obj)
        )
        ctype = np.ctypeslib.as_ctypes_type(self.dtype)
        arr_type = ctype * size
        c_arr = arr_type.from_address(addr)
        return np.ctypeslib.as_array(c_arr)

    @property
    def shape(self) -> tuple:
        return (self._cpp_obj.size(),)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, item, val):
        self.data[item] = val

    def __len__(self) -> int:
        return self._cpp_obj.size()


class CppyyDynamicArray2D:
    """
    cppyy-backed 2D dynamic array. API-compatible with Cython DynamicArray2DClass.
    """

    def __init__(self, shape: tuple, dtype=np.float64, factor: float = 2.0):
        _ensure_initialized()
        self.dtype = np.dtype(dtype)
        self._factor = factor

        rows = shape[0] if len(shape) > 0 else 0
        cols = shape[1] if len(shape) > 1 else 0

        cpp_type = _DTYPE_TO_CPP.get(self.dtype)
        if cpp_type is None:
            raise TypeError(f"Unsupported dtype: {self.dtype}")

        cls = getattr(_cppyy.gbl, f"DynamicArray2D<{cpp_type}>")
        self._cpp_obj = cls(int(rows), int(cols), float(factor))
        self._cpp_type = cpp_type

    def resize(self, new_shape) -> None:
        if (
            isinstance(new_shape, (tuple, list))
            and new_shape[1] != self._cpp_obj.cols()
        ):
            raise ValueError("Resizing is only supported along the first dimension")
        self.resize_along_first(new_shape)

    def resize_along_first(self, new_shape) -> None:
        if isinstance(new_shape, int):
            new_rows = new_shape
        elif isinstance(new_shape, (tuple, list)):
            new_rows = new_shape[0]
        else:
            raise ValueError("new_shape must be int, tuple, or list")
        self._cpp_obj.resize_along_first(int(new_rows))

    def shrink(self, new_shape) -> None:
        if isinstance(new_shape, int):
            self._cpp_obj.shrink(int(new_shape))
        else:
            self._cpp_obj.shrink(int(new_shape[0]), int(new_shape[1]))

    def get_capsule(self) -> Any:
        """Return PyCapsule wrapping the C++ DynamicArray2D<T>* pointer."""
        addr = _cppyy.addressof(self._cpp_obj)
        return _PyCapsule_New(addr, b"DynamicArray2D", None)

    @property
    def data(self) -> np.ndarray:
        """Zero-copy numpy 2D view with correct strides."""
        rows = self._cpp_obj.rows()
        cols = self._cpp_obj.cols()

        if rows == 0 or cols == 0:
            return np.zeros((rows, cols), dtype=self.dtype)

        addr = int(
            _cppyy.gbl._brian_dynarray_data_addr_2d[self._cpp_type](self._cpp_obj)
        )
        stride = self._cpp_obj.stride()  # physical row width (elements)
        itemsize = self.dtype.itemsize

        # Create 1D ctypes buffer covering the full physical layout
        ctype = np.ctypeslib.as_ctypes_type(self.dtype)
        total_elements = stride * rows
        arr_type = ctype * total_elements
        c_arr = arr_type.from_address(addr)
        flat = np.ctypeslib.as_array(c_arr)

        # Reshape with strides to get proper 2D view
        return np.lib.stride_tricks.as_strided(
            flat,
            shape=(rows, cols),
            strides=(stride * itemsize, itemsize),
        )

    @property
    def shape(self) -> tuple:
        return (self._cpp_obj.rows(), self._cpp_obj.cols())

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, item, val):
        self.data[item] = val

    def __len__(self) -> int:
        return self._cpp_obj.rows()


# Factory functions matching the Cython module's API exactly
def DynamicArray(shape, dtype=float, factor=2, use_numpy_resize=False, refcheck=True):
    if isinstance(shape, int):
        shape = (shape,)
    if len(shape) == 1:
        return CppyyDynamicArray1D(shape[0], dtype, factor)
    elif len(shape) == 2:
        return CppyyDynamicArray2D(shape, dtype, factor)
    else:
        raise ValueError(
            f"DynamicArray only supports 1D or 2D shapes. Got shape={shape}"
        )


def DynamicArray1D(shape, dtype=float, factor=2, use_numpy_resize=False, refcheck=True):
    if isinstance(shape, int):
        shape = (shape,)
    return CppyyDynamicArray1D(shape[0], dtype, factor)
