# cython: language_level=3
# distutils: language = c++
# distutils: sources = brian2/memory/cpp_standalone/cdynamicarray.cpp

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.string cimport memcpy
import cython
from cython.operator cimport dereference
cimport numpy as np
import numpy as np

np.import_array()


ctypedef fused scalar_type:
    np.float32_t
    np.float64_t
    np.int32_t
    np.int64_t
    np.int8_t
    np.uint8_t
    np.uint32_t
    np.uint64_t

# External C++ class declarations
cdef extern from "cdynamicarray.h":
    cdef cppclass CDynamicArray[T]:
        CDynamicArray(vector[size_t] shape, double factor) except +
        CDynamicArray(size_t size, double factor) except +
        T* data()
        vector[size_t] shape()
        vector[size_t] strides()
        size_t ndim()
        size_t size()
        void resize(vector[size_t] new_shape)
        void resize_1d(size_t new_size)
        void shrink(vector[size_t] new_shape)
        void get_slice(T* output, vector[pair[int, int]] slices)
        void set_slice(T* input, vector[pair[int, int]] slices)

    cdef cppclass CDynamicArray1D[T]:
        CDynamicArray1D(size_t size, double factor) except +
        T* data()
        size_t size()
        void resize(size_t new_size)
        void shrink(size_t new_size)

# Base class for dynamic arrays
cdef class DynamicArrayBase:
    cdef readonly np.dtype dtype
    cdef readonly tuple shape_tuple
    cdef readonly int ndim

    def __init__(self, shape, dtype):
        self.dtype = np.dtype(dtype)
        if isinstance(shape, int):
            self.shape_tuple = (shape,)
        else:
            self.shape_tuple = tuple(shape)
        self.ndim = len(self.shape_tuple)

# 1D Dynamic Array wrapper
cdef class DynamicArray1D(DynamicArrayBase):
    # Store pointers for different types
    cdef CDynamicArray1D[np.float32_t]* ptr_float32
    cdef CDynamicArray1D[np.float64_t]* ptr_float64
    cdef CDynamicArray1D[np.int32_t]* ptr_int32
    cdef CDynamicArray1D[np.int64_t]* ptr_int64
    cdef CDynamicArray1D[np.int8_t]* ptr_int8
    cdef CDynamicArray1D[np.uint8_t]* ptr_uint8
    cdef CDynamicArray1D[np.uint32_t]* ptr_uint32
    cdef CDynamicArray1D[np.uint64_t]* ptr_uint64

    cdef double factor

    def __cinit__(self, size, dtype=np.float64, factor=2.0):
        self.factor = factor
        self.ptr_float32 = NULL
        self.ptr_float64 = NULL
        self.ptr_int32 = NULL
        self.ptr_int64 = NULL
        self.ptr_int8 = NULL
        self.ptr_uint8 = NULL
        self.ptr_uint32 = NULL
        self.ptr_uint64 = NULL

    def __init__(self, size, dtype=np.float64, factor=2.0):
        super().__init__(size, dtype)

        # Create the appropriate C++ object based on dtype
        if self.dtype == np.float32:
            self.ptr_float32 = new CDynamicArray1D[np.float32_t](size, factor)
        elif self.dtype == np.float64:
            self.ptr_float64 = new CDynamicArray1D[np.float64_t](size, factor)
        elif self.dtype == np.int32:
            self.ptr_int32 = new CDynamicArray1D[np.int32_t](size, factor)
        elif self.dtype == np.int64:
            self.ptr_int64 = new CDynamicArray1D[np.int64_t](size, factor)
        elif self.dtype == np.int8:
            self.ptr_int8 = new CDynamicArray1D[np.int8_t](size, factor)
        elif self.dtype == np.uint8:
            self.ptr_uint8 = new CDynamicArray1D[np.uint8_t](size, factor)
        elif self.dtype == np.uint32:
            self.ptr_uint32 = new CDynamicArray1D[np.uint32_t](size, factor)
        elif self.dtype == np.uint64:
            self.ptr_uint64 = new CDynamicArray1D[np.uint64_t](size, factor)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

    def __dealloc__(self):
        if self.ptr_float32 != NULL:
            del self.ptr_float32
        if self.ptr_float64 != NULL:
            del self.ptr_float64
        if self.ptr_int32 != NULL:
            del self.ptr_int32
        if self.ptr_int64 != NULL:
            del self.ptr_int64
        if self.ptr_int8 != NULL:
            del self.ptr_int8
        if self.ptr_uint8 != NULL:
            del self.ptr_uint8
        if self.ptr_uint32 != NULL:
            del self.ptr_uint32
        if self.ptr_uint64 != NULL:
            del self.ptr_uint64

    @property
    def data(self):
        """Return a numpy array view of the data"""
        cdef np.npy_intp shape[1]
        cdef void* data_ptr

        if self.dtype == np.float32:
            shape[0] = self.ptr_float32.size()
            data_ptr = <void*>self.ptr_float32.data()
        elif self.dtype == np.float64:
            shape[0] = self.ptr_float64.size()
            data_ptr = <void*>self.ptr_float64.data()
        elif self.dtype == np.int32:
            shape[0] = self.ptr_int32.size()
            data_ptr = <void*>self.ptr_int32.data()
        elif self.dtype == np.int64:
            shape[0] = self.ptr_int64.size()
            data_ptr = <void*>self.ptr_int64.data()
        elif self.dtype == np.int8:
            shape[0] = self.ptr_int8.size()
            data_ptr = <void*>self.ptr_int8.data()
        elif self.dtype == np.uint8:
            shape[0] = self.ptr_uint8.size()
            data_ptr = <void*>self.ptr_uint8.data()
        elif self.dtype == np.uint32:
            shape[0] = self.ptr_uint32.size()
            data_ptr = <void*>self.ptr_uint32.data()
        elif self.dtype == np.uint64:
            shape[0] = self.ptr_uint64.size()
            data_ptr = <void*>self.ptr_uint64.data()

        # Create numpy array without copying
        return np.PyArray_SimpleNewFromData(1, shape, self.dtype.num, data_ptr)

    @property
    def shape(self):
        if self.dtype == np.float32:
            return (self.ptr_float32.size(),)
        elif self.dtype == np.float64:
            return (self.ptr_float64.size(),)
        elif self.dtype == np.int32:
            return (self.ptr_int32.size(),)
        elif self.dtype == np.int64:
            return (self.ptr_int64.size(),)
        elif self.dtype == np.int8:
            return (self.ptr_int8.size(),)
        elif self.dtype == np.uint8:
            return (self.ptr_uint8.size(),)
        elif self.dtype == np.uint32:
            return (self.ptr_uint32.size(),)
        elif self.dtype == np.uint64:
            return (self.ptr_uint64.size(),)

    def resize(self, newsize):
        """Resize the array"""
        if isinstance(newsize, tuple):
            newsize = newsize[0]

        if self.dtype == np.float32:
            self.ptr_float32.resize(newsize)
        elif self.dtype == np.float64:
            self.ptr_float64.resize(newsize)
        elif self.dtype == np.int32:
            self.ptr_int32.resize(newsize)
        elif self.dtype == np.int64:
            self.ptr_int64.resize(newsize)
        elif self.dtype == np.int8:
            self.ptr_int8.resize(newsize)
        elif self.dtype == np.uint8:
            self.ptr_uint8.resize(newsize)
        elif self.dtype == np.uint32:
            self.ptr_uint32.resize(newsize)
        elif self.dtype == np.uint64:
            self.ptr_uint64.resize(newsize)

        self.shape_tuple = (newsize,)

    def shrink(self, newsize):
        """Shrink the array, deallocating extra memory"""
        if isinstance(newsize, tuple):
            newsize = newsize[0]

        if self.dtype == np.float32:
            self.ptr_float32.shrink(newsize)
        elif self.dtype == np.float64:
            self.ptr_float64.shrink(newsize)
        elif self.dtype == np.int32:
            self.ptr_int32.shrink(newsize)
        elif self.dtype == np.int64:
            self.ptr_int64.shrink(newsize)
        elif self.dtype == np.int8:
            self.ptr_int8.shrink(newsize)
        elif self.dtype == np.uint8:
            self.ptr_uint8.shrink(newsize)
        elif self.dtype == np.uint32:
            self.ptr_uint32.shrink(newsize)
        elif self.dtype == np.uint64:
            self.ptr_uint64.shrink(newsize)

        self.shape_tuple = (newsize,)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"DynamicArray1D(shape={self.shape}, dtype={self.dtype})"


# Multi-dimensional Dynamic Array wrapper
cdef class DynamicArray(DynamicArrayBase):
    # Store pointers for different types
    cdef CDynamicArray[np.float32_t]* ptr_float32
    cdef CDynamicArray[np.float64_t]* ptr_float64
    cdef CDynamicArray[np.int32_t]* ptr_int32
    cdef CDynamicArray[np.int64_t]* ptr_int64
    cdef CDynamicArray[np.int8_t]* ptr_int8
    cdef CDynamicArray[np.uint8_t]* ptr_uint8
    cdef CDynamicArray[np.uint32_t]* ptr_uint32
    cdef CDynamicArray[np.uint64_t]* ptr_uint64

    cdef double factor

    def __cinit__(self, shape, dtype=np.float64, factor=2.0):
        self.factor = factor
        self.ptr_float32 = NULL
        self.ptr_float64 = NULL
        self.ptr_int32 = NULL
        self.ptr_int64 = NULL
        self.ptr_int8 = NULL
        self.ptr_uint8 = NULL
        self.ptr_uint32 = NULL
        self.ptr_uint64 = NULL

    def __init__(self, shape, dtype=np.float64, factor=2.0):
        super().__init__(shape, dtype)

        cdef vector[size_t] cpp_shape
        for dim in self.shape_tuple:
            cpp_shape.push_back(dim)

        # Create the appropriate C++ object based on dtype
        if self.dtype == np.float32:
            self.ptr_float32 = new CDynamicArray[np.float32_t](cpp_shape, factor)
        elif self.dtype == np.float64:
            self.ptr_float64 = new CDynamicArray[np.float64_t](cpp_shape, factor)
        elif self.dtype == np.int32:
            self.ptr_int32 = new CDynamicArray[np.int32_t](cpp_shape, factor)
        elif self.dtype == np.int64:
            self.ptr_int64 = new CDynamicArray[np.int64_t](cpp_shape, factor)
        elif self.dtype == np.int8:
            self.ptr_int8 = new CDynamicArray[np.int8_t](cpp_shape, factor)
        elif self.dtype == np.uint8:
            self.ptr_uint8 = new CDynamicArray[np.uint8_t](cpp_shape, factor)
        elif self.dtype == np.uint32:
            self.ptr_uint32 = new CDynamicArray[np.uint32_t](cpp_shape, factor)
        elif self.dtype == np.uint64:
            self.ptr_uint64 = new CDynamicArray[np.uint64_t](cpp_shape, factor)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

    def __dealloc__(self):
        if self.ptr_float32 != NULL:
            del self.ptr_float32
        if self.ptr_float64 != NULL:
            del self.ptr_float64
        if self.ptr_int32 != NULL:
            del self.ptr_int32
        if self.ptr_int64 != NULL:
            del self.ptr_int64
        if self.ptr_int8 != NULL:
            del self.ptr_int8
        if self.ptr_uint8 != NULL:
            del self.ptr_uint8
        if self.ptr_uint32 != NULL:
            del self.ptr_uint32
        if self.ptr_uint64 != NULL:
            del self.ptr_uint64

    @property
    def data(self):
        """Return a numpy array view of the data"""
        cdef np.npy_intp* shape_arr
        cdef np.npy_intp* strides_arr
        cdef void* data_ptr
        cdef vector[size_t] cpp_shape
        cdef vector[size_t] cpp_strides
        cdef int itemsize = self.dtype.itemsize

        if self.dtype == np.float32:
            cpp_shape = self.ptr_float32.shape()
            cpp_strides = self.ptr_float32.strides()
            data_ptr = <void*>self.ptr_float32.data()
        elif self.dtype == np.float64:
            cpp_shape = self.ptr_float64.shape()
            cpp_strides = self.ptr_float64.strides()
            data_ptr = <void*>self.ptr_float64.data()
        elif self.dtype == np.int32:
            cpp_shape = self.ptr_int32.shape()
            cpp_strides = self.ptr_int32.strides()
            data_ptr = <void*>self.ptr_int32.data()
        elif self.dtype == np.int64:
            cpp_shape = self.ptr_int64.shape()
            cpp_strides = self.ptr_int64.strides()
            data_ptr = <void*>self.ptr_int64.data()
        elif self.dtype == np.int8:
            cpp_shape = self.ptr_int8.shape()
            cpp_strides = self.ptr_int8.strides()
            data_ptr = <void*>self.ptr_int8.data()
        elif self.dtype == np.uint8:
            cpp_shape = self.ptr_uint8.shape()
            cpp_strides = self.ptr_uint8.strides()
            data_ptr = <void*>self.ptr_uint8.data()
        elif self.dtype == np.uint32:
            cpp_shape = self.ptr_uint32.shape()
            cpp_strides = self.ptr_uint32.strides()
            data_ptr = <void*>self.ptr_uint32.data()
        elif self.dtype == np.uint64:
            cpp_shape = self.ptr_uint64.shape()
            cpp_strides = self.ptr_uint64.strides()
            data_ptr = <void*>self.ptr_uint64.data()

        # Convert shape and strides to numpy format
        shape_arr = <np.npy_intp*>np.PyMem_Malloc(self.ndim * sizeof(np.npy_intp))
        strides_arr = <np.npy_intp*>np.PyMem_Malloc(self.ndim * sizeof(np.npy_intp))

        for i in range(self.ndim):
            shape_arr[i] = cpp_shape[i]
            strides_arr[i] = cpp_strides[i] * itemsize

        # Create numpy array without copying
        cdef np.ndarray arr = np.PyArray_New(
            np.ndarray,
            self.ndim,
            shape_arr,
            self.dtype.num,
            strides_arr,
            data_ptr,
            itemsize,
            np.NPY_ARRAY_CARRAY,
            None
        )

        # The array now owns these arrays
        np.PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)

        return arr

    @property
    def shape(self):
        cdef vector[size_t] cpp_shape

        if self.dtype == np.float32:
            cpp_shape = self.ptr_float32.shape()
        elif self.dtype == np.float64:
            cpp_shape = self.ptr_float64.shape()
        elif self.dtype == np.int32:
            cpp_shape = self.ptr_int32.shape()
        elif self.dtype == np.int64:
            cpp_shape = self.ptr_int64.shape()
        elif self.dtype == np.int8:
            cpp_shape = self.ptr_int8.shape()
        elif self.dtype == np.uint8:
            cpp_shape = self.ptr_uint8.shape()
        elif self.dtype == np.uint32:
            cpp_shape = self.ptr_uint32.shape()
        elif self.dtype == np.uint64:
            cpp_shape = self.ptr_uint64.shape()

        return tuple(cpp_shape)

    def resize(self, newshape):
        """Resize the array"""
        if isinstance(newshape, int):
            newshape = (newshape,)

        cdef vector[size_t] cpp_shape
        for dim in newshape:
            cpp_shape.push_back(dim)

        if self.dtype == np.float32:
            self.ptr_float32.resize(cpp_shape)
        elif self.dtype == np.float64:
            self.ptr_float64.resize(cpp_shape)
        elif self.dtype == np.int32:
            self.ptr_int32.resize(cpp_shape)
        elif self.dtype == np.int64:
            self.ptr_int64.resize(cpp_shape)
        elif self.dtype == np.int8:
            self.ptr_int8.resize(cpp_shape)
        elif self.dtype == np.uint8:
            self.ptr_uint8.resize(cpp_shape)
        elif self.dtype == np.uint32:
            self.ptr_uint32.resize(cpp_shape)
        elif self.dtype == np.uint64:
            self.ptr_uint64.resize(cpp_shape)

        self.shape_tuple = tuple(newshape)

    def shrink(self, newshape):
        """Shrink the array, deallocating extra memory"""
        if isinstance(newshape, int):
            newshape = (newshape,)

        cdef vector[size_t] cpp_shape
        for dim in newshape:
            cpp_shape.push_back(dim)

        if self.dtype == np.float32:
            self.ptr_float32.shrink(cpp_shape)
        elif self.dtype == np.float64:
            self.ptr_float64.shrink(cpp_shape)
        elif self.dtype == np.int32:
            self.ptr_int32.shrink(cpp_shape)
        elif self.dtype == np.int64:
            self.ptr_int64.shrink(cpp_shape)
        elif self.dtype == np.int8:
            self.ptr_int8.shrink(cpp_shape)
        elif self.dtype == np.uint8:
            self.ptr_uint8.shrink(cpp_shape)
        elif self.dtype == np.uint32:
            self.ptr_uint32.shrink(cpp_shape)
        elif self.dtype == np.uint64:
            self.ptr_uint64.shrink(cpp_shape)

        self.shape_tuple = tuple(newshape)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return str(self.data)
