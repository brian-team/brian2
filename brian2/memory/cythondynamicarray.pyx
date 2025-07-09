# cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3
# distutils: language = c++
# distutils: include_dirs = brian2/devices/cpp_standalone/brianlib
# # distutils: extra_compile_args = -std=c++11

import numpy as np
cimport numpy as cnp
cimport cython
from cythondynamicarray cimport DynamicArray1D, DynamicArray2D
from libc.string cimport memset
from cython cimport view

cnp.import_array()


cdef extern from "dynamic_array.h"
    cdef cppclass DynamicArray1D[T]:
        DynamicArray1D(size_t,double) except +
        void resize(size_t) except +
        void shrink_to_fit()
        T& operator[](size_t)
        T* get_data_ptr()
        size_t size()
        size_t capacity()

    cdef cppclass DynamicArray2D[T]:
        size_t n  # rows
        size_t m  # cols
        DynamicArray2D(size_t, size_t, double) except +
        DynamicArray2D(int, int) except + # Legacy constructor
        void resize(size_t, size_t) except +
        void resize(int, int) except + # Legacy method
        void resize() except +
        void shrink_to_fit()
        T& operator()(size_t, size_t)
        T& operator()(int, int)
        T* get_data_ptr()
        size_t rows()
        size_t cols()
        size_t stride()


# Fused type for numeric types
ctypedef fused numeric:
    double
    float
    int
    long
    cython.bint

# We have to define a mapping for numpy dtypes to our class
cdef dict NUMPY_TYPE_MAP = {
    np.float64: cnp.NPY_DOUBLE,
    np.float32: cnp.NPY_FLOAT,
    np.int32: cnp.NPY_INT32,
    np.int64: cnp.NPY_INT64,
    np.bool_: cnp.NPY_BOOL
}


cdef class DynamicArray1D:
    cdef void* thisptr
    cdef int NUMPY_TYPE_MAP
    cdef object dtype
    cdef double factor

    def __cint__(self,size_t intial_size, dtype = np.float64, double factor=2.0):
        self.dtype = np.dtype(dtype)
        self.factor = factor
        self.numpy_type = NUMPY_TYPE_MAP[self.dtype.type]

        if self.dtype == np.float64:
            self.thisptr = <void*>DynamicArray1D[double](intial_size,factor)
        elif self.dtype == np.float32:
            self.thisptr = <void*>DynamicArray1D[float](intial_size,factor)
        elif self.dtype == np.int32:
            self.thisptr = <void*>DynamicArray1D[int](intial_size,factor)
        elif self.dtype == np.int64:
            self.thisptr = <void*>DynamicArray1D[long](intial_size,factor)
        elif self.dtype == np.bool_:
            self.thisptr = <void*>DynamicArray1D[cython.bint](intial_size,factor)
        else:
            raise TypeError("Unsupported dtype: {}".format(self.dtype))

    def __dealloc__(self):
        if self.thisptr != NULL:
            if self.dtype == np.float64:
                del <DynamicArray1D[double]*>self.thisptr
            elif self.dtype == np.float32:
                del <DynamicArray1D[float]*>self.thisptr
            elif self.dtype == np.int32:
                del <DynamicArray1D[int]*>self.thisptr
            elif self.dtype == np.int64:
                del <DynamicArray1D[long]*>self.thisptr
            elif self.dtype == np.bool_:
                del <DynamicArray1D[cython.bint]*>self.thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void* get_data_ptr(self) noexcept nogil:
        """C-level access to data pointer"""
        if self.dtype == np.float64:
            return <void*>(<DynamicArray1D[double]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.float32:
            return <void*>(<DynamicArray1D[float]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.int32:
            return <void*>(<DynamicArray1D[int]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.int64:
            return <void*>(<DynamicArray1D[long]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.bool_:
            return <void*>(<DynamicArray1D[cython.bint]*>self.thisptr).get_data_ptr()
        return NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef size_t get_size(self) noexcept nogil:
        """C-level access to size"""
        if self.dtype == np.float64:
            return (<DynamicArray1D[double]*>self.thisptr).size()
        elif self.dtype == np.float32:
            return (<DynamicArray1D[float]*>self.thisptr).size()
        elif self.dtype == np.int32:
            return (<DynamicArray1D[int]*>self.thisptr).size()
        elif self.dtype == np.int64:
            return (<DynamicArray1D[long]*>self.thisptr).size()
        elif self.dtype == np.bool_:
            return (<DynamicArray1D[cython.bint]*>self.thisptr).size()
        return 0

    def resize(self, size_t new_size):
        """Resize array to new size"""
        if self.dtype == np.float64:
            (<DynamicArray1D[double]*>self.thisptr).resize(new_size)
        elif self.dtype == np.float32:
            (<DynamicArray1D[float]*>self.thisptr).resize(new_size)
        elif self.dtype == np.int32:
            (<DynamicArray1D[int]*>self.thisptr).resize(new_size)
        elif self.dtype == np.int64:
            (<DynamicArray1D[long]*>self.thisptr).resize(new_size)
        elif self.dtype == np.bool_:
            (<DynamicArray1D[cython.bint]*>self.thisptr).resize(new_size)

    @property
    def data(self):
        """Return numpy array view of underlying data"""
        cdef cnp.npy_intp shape[1]
        cdef size_t size = self.get_size()
        cdef void* data_ptr = self.get_data_ptr()

        shape[0] = size
        if size == 0:
            return np.array([], dtype=self.dtype)
        # Note : This creates a zero-copy NumPy view over the memory allocated by the C++ backend —
        # changes to the NumPy array will reflect in the C++ array and vice versa.
        return cnp.PyArray_SimpleNewFromData(1, shape, self.numpy_type, data_ptr)

    @property
    def shape(self):
        return (self.get_size(),)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, item, val):
        cdef cnp.ndarray arr = self.data
        arr[item] = val

    def __len__(self):
        return self.get_size()


cdef class DynamicArray2D:
    cdef void* thisptr
    cdef int numpy_type
    cdef object dtype
    cdef double factor

    def __cinit__(self, tuple shape, dtype=np.float64, double factor=2.0):
        cdef size_t rows = shape[0] if len(shape) > 0 else 0
        cdef size_t cols = shape[1] if len(shape) > 1 else 0

        self.dtype = np.dtype(dtype)
        self.factor = factor
        self.numpy_type = NUMPY_TYPE_MAP[self.dtype.type]

        if self.dtype == np.float64:
            self.thisptr = new DynamicArray2D[double](rows, cols, factor)
        elif self.dtype == np.float32:
            self.thisptr = new DynamicArray2D[float](rows, cols, factor)
        elif self.dtype == np.int32:
            self.thisptr = new DynamicArray2D[int](rows, cols, factor)
        elif self.dtype == np.int64:
            self.thisptr = new DynamicArray2D[long](rows, cols, factor)
        elif self.dtype == np.bool_:
            self.thisptr = new DynamicArray2D[cython.bint](rows, cols, factor)
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")

    def __dealloc__(self):
        if self.thisptr != NULL:
            if self.dtype == np.float64:
                del <DynamicArray2D[double]*>self.thisptr
            elif self.dtype == np.float32:
                del <DynamicArray2D[float]*>self.thisptr
            elif self.dtype == np.int32:
                del <DynamicArray2D[int]*>self.thisptr
            elif self.dtype == np.int64:
                del <DynamicArray2D[long]*>self.thisptr
            elif self.dtype == np.bool_:
                del <DynamicArray2D[cython.bint]*>self.thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void* get_data_ptr(self) noexcept nogil:
        """C-level access to data pointer"""
        if self.dtype == np.float64:
            return <void*>(<DynamicArray2D[double]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.float32:
            return <void*>(<DynamicArray2D[float]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.int32:
            return <void*>(<DynamicArray2D[int]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.int64:
            return <void*>(<DynamicArray2D[long]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.bool_:
            return <void*>(<DynamicArray2D[cython.bint]*>self.thisptr).get_data_ptr()
        return NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef size_t get_rows(self) noexcept nogil:
        """C-level access to rows"""
        if self.dtype == np.float64:
            return (<DynamicArray2D[double]*>self.thisptr).rows()
        elif self.dtype == np.float32:
            return (<DynamicArray2D[float]*>self.thisptr).rows()
        elif self.dtype == np.int32:
            return (<DynamicArray2D[int]*>self.thisptr).rows()
        elif self.dtype == np.int64:
            return (<DynamicArray2D[long]*>self.thisptr).rows()
        elif self.dtype == np.bool_:
            return (<DynamicArray2D[cython.bint]*>self.thisptr).rows()
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef size_t get_cols(self) noexcept nogil:
        """C-level access to cols"""
        if self.dtype == np.float64:
            return (<DynamicArray2D[double]*>self.thisptr).cols()
        elif self.dtype == np.float32:
            return (<DynamicArray2D[float]*>self.thisptr).cols()
        elif self.dtype == np.int32:
            return (<DynamicArray2D[int]*>self.thisptr).cols()
        elif self.dtype == np.int64:
            return (<DynamicArray2D[long]*>self.thisptr).cols()
        elif self.dtype == np.bool_:
            return (<DynamicArray2D[cython.bint]*>self.thisptr).cols()
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef size_t get_stride(self) noexcept nogil:
        """C-level access to stride"""
        if self.dtype == np.float64:
            return (<DynamicArray2D[double]*>self.thisptr).stride()
        elif self.dtype == np.float32:
            return (<DynamicArray2D[float]*>self.thisptr).stride()
        elif self.dtype == np.int32:
            return (<DynamicArray2D[int]*>self.thisptr).stride()
        elif self.dtype == np.int64:
            return (<DynamicArray2D[long]*>self.thisptr).stride()
        elif self.dtype == np.bool_:
            return (<DynamicArray2D[cython.bint]*>self.thisptr).stride()
        return 0

    def resize(self, tuple new_shape):
        """Resize array to new shape"""
        cdef size_t new_rows = new_shape[0]
        cdef size_t new_cols = new_shape[1]

        if self.dtype == np.float64:
            (<DynamicArray2D[double]*>self.thisptr).resize(new_rows, new_cols)
        elif self.dtype == np.float32:
            (<DynamicArray2D[float]*>self.thisptr).resize(new_rows, new_cols)
        elif self.dtype == np.int32:
            (<DynamicArray2D[int]*>self.thisptr).resize(new_rows, new_cols)
        elif self.dtype == np.int64:
            (<DynamicArray2D[long]*>self.thisptr).resize(new_rows, new_cols)
        elif self.dtype == np.bool_:
            (<DynamicArray2D[cython.bint]*>self.thisptr).resize(new_rows, new_cols)

    @property
    def data(self):
        """Return numpy array view with proper strides"""
        cdef cnp.npy_intp shape[2]
        cdef cnp.npy_intp strides[2]
        cdef size_t rows = self.get_rows()
        cdef size_t cols = self.get_cols()
        cdef size_t stride = self.get_stride()
        cdef void* data_ptr = self.get_data_ptr()
        cdef size_t itemsize = self.dtype.itemsize

        if rows == 0 or cols == 0:
            return np.array([], dtype=self.dtype).reshape((0, 0))

        shape[0] = rows
        shape[1] = cols
        strides[0] = stride * itemsize
        strides[1] = itemsize

        return cnp.PyArray_NewFromDescr(
            cnp.ndarray, self.dtype, 2, shape, strides, data_ptr, 0, None)

    @property
    def shape(self):
        return (self.get_rows(), self.get_cols())

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, item, val):
        cdef cnp.ndarray arr = self.data
        arr[item] = val

    def __len__(self):
        return self.get_rows()


# Factory functions matching original API we had in python code
def DynamicArray(shape, dtype=float, factor=2, use_numpy_resize=False, refcheck=True):
    """Create appropriate dynamic array based on shape"""
    if isinstance(shape, int):
        shape = (shape,)

    if len(shape) == 1:
        return DynamicArray1D(shape[0], dtype, factor)
    elif len(shape) == 2:
        return DynamicArray2D(shape, dtype, factor)
    else:
        # Flatten higher dimensions to 2D
        flat_shape = (int(np.prod(shape[:-1])), shape[-1])
        return FastDynamicArray2D(flat_shape, dtype, factor)

def DynamicArray1D(shape, dtype=float, factor=2, use_numpy_resize=False, refcheck=True):
    """Create 1D dynamic array"""
    if isinstance(shape, int):
        shape = (shape,)
    return DynamicArray1D(shape[0], dtype, factor)
