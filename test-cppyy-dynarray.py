"""Test cppyy DynamicArray implementation."""
import numpy as np

# Test the cppyy implementation directly
from brian2.memory.cppyy_dynamicarray import (
    CppyyDynamicArray1D,
    CppyyDynamicArray2D,
    DynamicArray,
    DynamicArray1D,
)

print("=" * 60)
print("TEST: CppyyDynamicArray1D")
print("=" * 60)

for dtype in [np.float64, np.float32, np.int32, np.int64, np.bool_]:
    arr = CppyyDynamicArray1D(10, dtype=dtype)
    assert len(arr) == 10, f"len={len(arr)}, expected 10"
    assert arr.shape == (10,), f"shape={arr.shape}"
    assert arr.data.dtype == dtype or (dtype == np.bool_ and arr.data.dtype == np.int8), f"dtype mismatch: {arr.data.dtype}"
    arr[0] = 42 if dtype != np.bool_ else 1
    assert arr[0] == (42 if dtype != np.bool_ else 1)
    arr.resize(20)
    assert len(arr) == 20
    arr.shrink(5)
    assert len(arr) == 5
    capsule = arr.get_capsule()
    assert capsule is not None
    print(f"  {dtype}: OK")

print()
print("=" * 60)
print("TEST: CppyyDynamicArray2D")
print("=" * 60)

for dtype in [np.float64, np.int32]:
    arr2d = CppyyDynamicArray2D((5, 3), dtype=dtype)
    assert arr2d.shape == (5, 3), f"shape={arr2d.shape}"
    assert len(arr2d) == 5
    d = arr2d.data
    assert d.shape == (5, 3), f"data shape={d.shape}"
    arr2d[0, 0] = 99
    assert arr2d[0, 0] == 99
    arr2d.resize_along_first(10)
    assert arr2d.shape == (10, 3)
    capsule = arr2d.get_capsule()
    assert capsule is not None
    print(f"  {dtype}: OK")

print()
print("=" * 60)
print("TEST: Factory functions")
print("=" * 60)

a1 = DynamicArray(5, dtype=np.float64)
assert isinstance(a1, CppyyDynamicArray1D)
a2 = DynamicArray((3, 4), dtype=np.int32)
assert isinstance(a2, CppyyDynamicArray2D)
a3 = DynamicArray1D(10, dtype=np.float32)
assert isinstance(a3, CppyyDynamicArray1D)
print("  Factory functions: OK")

print()
print("=" * 60)
print("TEST: Capsule compatibility with cppyy C++ extraction")
print("=" * 60)

import cppyy
# Ensure support code is loaded
from brian2.codegen.runtime.cppyy_rt.cppyy_rt import _ensure_support_code
_ensure_support_code()

arr = CppyyDynamicArray1D(5, dtype=np.float64)
arr[:] = [1.0, 2.0, 3.0, 4.0, 5.0]
capsule = arr.get_capsule()

# Extract the C++ pointer from capsule in C++ and verify data
cppyy.cppdef("""
extern "C" double _test_capsule_extract(PyObject* cap) {
    auto* dyn = _extract_dynamic_array_1d<double>(cap);
    return dyn->get_data_ptr()[2];  // should be 3.0
}
""")
result = cppyy.gbl._test_capsule_extract(capsule)
assert result == 3.0, f"Expected 3.0, got {result}"
print("  Capsule extraction: OK (C++ read value 3.0)")

# Test resize from C++ side
cppyy.cppdef("""
extern "C" void _test_capsule_resize(PyObject* cap) {
    auto* dyn = _extract_dynamic_array_1d<double>(cap);
    dyn->resize(10);
    dyn->get_data_ptr()[9] = 99.0;
}
""")
cppyy.gbl._test_capsule_resize(capsule)
assert len(arr) == 10, f"len={len(arr)}, expected 10 after C++ resize"
assert arr[9] == 99.0, f"arr[9]={arr[9]}, expected 99.0"
print("  C++ resize via capsule: OK")

print()
print("All DynamicArray tests passed!")
