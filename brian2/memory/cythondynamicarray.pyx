# cython: language_level=3
# distutils: language = c++
# distutils: include_dirs = brian2/devices/cpp_standalone/brianlib
# distutils: extra_compile_args = -std=c++11


import numpy as np
cimport numpy as cnp
cimport cython
from libc.string cimport memset
from libc.stdint cimport int64_t, int32_t
from cython cimport view
from cpython.pycapsule cimport PyCapsule_New
from cpython.ref cimport PyTypeObject

cnp.import_array()

cdef extern from "numpy/ndarrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype,
                                cnp.PyArray_Descr* descr,
                                int nd,
                                cnp.npy_intp* dims,
                                cnp.npy_intp* strides,
                                void* data,
                                int flags,
                                object obj)
    cnp.PyArray_Descr* PyArray_DescrFromType(int)

cdef extern from "numpy/ndarraytypes.h":
    void PyArray_CLEARFLAGS(cnp.PyArrayObject *arr, int flags)
    enum:
        NPY_ARRAY_C_CONTIGUOUS
        NPY_ARRAY_F_CONTIGUOUS
        NPY_ARRAY_OWNDATA
        NPY_ARRAY_WRITEABLE
        NPY_ARRAY_ALIGNED
        NPY_ARRAY_WRITEBACKIFCOPY
        NPY_ARRAY_UPDATEIFCOPY

cdef extern from "dynamic_array.h":
    cdef cppclass DynamicArray1DCpp "DynamicArray1D"[T]:
        DynamicArray1DCpp(size_t,double) except +
        void resize(size_t) except +
        void shrink_to_fit()
        void shrink(size_t) except +
        T& operator[](size_t)
        T* get_data_ptr()
        size_t size()
        size_t capacity()


    cdef cppclass DynamicArray2DCpp "DynamicArray2D"[T]:
        size_t n  # rows
        size_t m  # cols
        DynamicArray2DCpp(size_t, size_t, double) except +
        DynamicArray2DCpp(int, int) except + # Legacy constructor
        void resize(size_t, size_t) except +
        void resize(int, int) except + # Legacy method
        void resize() except +
        void resize_along_first(size_t) except +
        void shrink_to_fit()
        void shrink(size_t, size_t) except +
        void shrink(size_t) except +
        T& operator()(size_t, size_t)
        T& operator()(int, int)
        T* get_data_ptr()
        size_t rows()
        size_t cols()
        size_t stride()


# We have to define a mapping for numpy dtypes to our class
cdef dict NUMPY_TYPE_MAP = {
    np.float64: cnp.NPY_DOUBLE,
    np.float32: cnp.NPY_FLOAT,
    np.int32: cnp.NPY_INT32,
    np.int64: cnp.NPY_INT64,
    np.bool_: cnp.NPY_BOOL
}


cdef class DynamicArray1DClass:
    cdef void* thisptr
    cdef int numpy_type
    cdef object dtype
    cdef double factor

    def __cinit__(self,size_t intial_size, dtype = np.float64, double factor=2.0):
        self.dtype = np.dtype(dtype)
        self.factor = factor
        self.numpy_type = NUMPY_TYPE_MAP[self.dtype.type]

        if self.dtype == np.float64:
            self.thisptr = new DynamicArray1DCpp[double](intial_size,factor)
        elif self.dtype == np.float32:
            self.thisptr = new DynamicArray1DCpp[float](intial_size,factor)
        elif self.dtype == np.int32:
            self.thisptr = new DynamicArray1DCpp[int32_t](intial_size,factor)
        elif self.dtype == np.int64:
            self.thisptr = new DynamicArray1DCpp[int64_t](intial_size,factor)
        elif self.dtype == np.bool_:
             # When asked for a bool array, we create a char array in C++ as for bool, C++ tries to optimize and save memory by packing all the boolean values tightly — 1 bit per value, instead of 1 byte.
            self.thisptr = new DynamicArray1DCpp[char](intial_size,factor)
        else:
            raise TypeError("Unsupported dtype: {}".format(self.dtype))

    def __dealloc__(self):
        cdef DynamicArray1DCpp[double]* ptr_double
        cdef DynamicArray1DCpp[float]* ptr_float
        cdef DynamicArray1DCpp[int32_t]* ptr_int
        cdef DynamicArray1DCpp[int64_t]* ptr_long
        cdef DynamicArray1DCpp[char]* ptr_bool
        if self.thisptr != NULL:
            if self.dtype == np.float64:
                ptr_double = <DynamicArray1DCpp[double]*>self.thisptr
                del ptr_double
            elif self.dtype == np.float32:
                ptr_float = <DynamicArray1DCpp[float]*>self.thisptr
                del ptr_float
            elif self.dtype == np.int32:
                ptr_int = <DynamicArray1DCpp[int32_t]*>self.thisptr
                del ptr_int
            elif self.dtype == np.int64:
                ptr_long = <DynamicArray1DCpp[int64_t]*>self.thisptr
                del ptr_long
            elif self.dtype == np.bool_:
                ptr_bool = <DynamicArray1DCpp[char]*>self.thisptr
                del ptr_bool


    cdef void* get_data_ptr(self) :
        """C-level access to data pointer"""
        if self.dtype == np.float64:
            return <void*>(<DynamicArray1DCpp[double]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.float32:
            return <void*>(<DynamicArray1DCpp[float]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.int32:
            return <void*>(<DynamicArray1DCpp[int32_t]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.int64:
            return <void*>(<DynamicArray1DCpp[int64_t]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.bool_:
            return <void*>(<DynamicArray1DCpp[char]*>self.thisptr).get_data_ptr()
        return NULL

    def get_capsule(self):
        """
        Returns a PyCapsule object wrapping the underlying C++ Dynamic1D Array class pointer.

        PyCapsules are used to safely pass raw C/C++ pointers between Python modules
        or C extensions without exposing the actual implementation details to Python.
        """
        return PyCapsule_New(<void*>self.thisptr, "DynamicArray1D", NULL)

    cdef size_t get_size(self):
        """C-level access to size"""
        if self.dtype == np.float64:
            return (<DynamicArray1DCpp[double]*>self.thisptr).size()
        elif self.dtype == np.float32:
            return (<DynamicArray1DCpp[float]*>self.thisptr).size()
        elif self.dtype == np.int32:
            return (<DynamicArray1DCpp[int32_t]*>self.thisptr).size()
        elif self.dtype == np.int64:
            return (<DynamicArray1DCpp[int64_t]*>self.thisptr).size()
        elif self.dtype == np.bool_:
            return (<DynamicArray1DCpp[char]*>self.thisptr).size()
        return 0

    def resize(self, size_t new_size):
        """Resize array to new size"""
        if self.dtype == np.float64:
            (<DynamicArray1DCpp[double]*>self.thisptr).resize(new_size)
        elif self.dtype == np.float32:
            (<DynamicArray1DCpp[float]*>self.thisptr).resize(new_size)
        elif self.dtype == np.int32:
            (<DynamicArray1DCpp[int32_t]*>self.thisptr).resize(new_size)
        elif self.dtype == np.int64:
            (<DynamicArray1DCpp[int64_t]*>self.thisptr).resize(new_size)
        elif self.dtype == np.bool_:
            (<DynamicArray1DCpp[char]*>self.thisptr).resize(new_size)

    def shrink(self, size_t new_size):
        """Shrink array to exact new size, freeing unused memory"""
        if self.dtype == np.float64:
            (<DynamicArray1DCpp[double]*>self.thisptr).shrink(new_size)
        elif self.dtype == np.float32:
            (<DynamicArray1DCpp[float]*>self.thisptr).shrink(new_size)
        elif self.dtype == np.int32:
            (<DynamicArray1DCpp[int32_t]*>self.thisptr).shrink(new_size)
        elif self.dtype == np.int64:
            (<DynamicArray1DCpp[int64_t]*>self.thisptr).shrink(new_size)
        elif self.dtype == np.bool_:
            (<DynamicArray1DCpp[char]*>self.thisptr).shrink(new_size)

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


cdef class DynamicArray2DClass:
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
            self.thisptr = new DynamicArray2DCpp[double](rows, cols, factor)
        elif self.dtype == np.float32:
            self.thisptr = new DynamicArray2DCpp[float](rows, cols, factor)
        elif self.dtype == np.int32:
            self.thisptr = new DynamicArray2DCpp[int32_t](rows, cols, factor)
        elif self.dtype == np.int64:
            self.thisptr = new DynamicArray2DCpp[int64_t](rows, cols, factor)
        elif self.dtype == np.bool_:
            # When asked for a bool array, we create a char array in C++ as for bool, C++ tries to optimize and save memory by packing all the boolean values tightly — 1 bit per value, instead of 1 byte.
            self.thisptr = new DynamicArray2DCpp[char](rows, cols, factor)
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")

    def __dealloc__(self):
        cdef DynamicArray2DCpp[double]* ptr_double
        cdef DynamicArray2DCpp[float]* ptr_float
        cdef DynamicArray2DCpp[int32_t]* ptr_int
        cdef DynamicArray2DCpp[int64_t]* ptr_long
        cdef DynamicArray2DCpp[char]* ptr_bool
        if self.thisptr != NULL:
            if self.dtype == np.float64:
                ptr_double = <DynamicArray2DCpp[double]*>self.thisptr
                del ptr_double
            elif self.dtype == np.float32:
                ptr_float = <DynamicArray2DCpp[float]*>self.thisptr
                del ptr_float
            elif self.dtype == np.int32:
                ptr_int = <DynamicArray2DCpp[int32_t]*>self.thisptr
                del ptr_int
            elif self.dtype == np.int64:
                ptr_long = <DynamicArray2DCpp[int64_t]*>self.thisptr
                del ptr_long
            elif self.dtype == np.bool_:
                ptr_bool = <DynamicArray2DCpp[char]*>self.thisptr
                del ptr_bool

    def get_capsule(self):
        """
        Returns a PyCapsule object wrapping the underlying C++ Dynamic1D Array class pointer.

        PyCapsules are used to safely pass raw C/C++ pointers between Python modules
        or C extensions without exposing the actual implementation details to Python.
        """
        return PyCapsule_New(<void*>self.thisptr, "DynamicArray2D", NULL)

    cdef void* get_data_ptr(self):
        """C-level access to data pointer"""
        if self.dtype == np.float64:
            return <void*>(<DynamicArray2DCpp[double]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.float32:
            return <void*>(<DynamicArray2DCpp[float]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.int32:
            return <void*>(<DynamicArray2DCpp[int32_t]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.int64:
            return <void*>(<DynamicArray2DCpp[int64_t]*>self.thisptr).get_data_ptr()
        elif self.dtype == np.bool_:
            return <void*>(<DynamicArray2DCpp[char]*>self.thisptr).get_data_ptr()
        return NULL

    cdef size_t get_rows(self):
        """C-level access to rows"""
        if self.dtype == np.float64:
            return (<DynamicArray2DCpp[double]*>self.thisptr).rows()
        elif self.dtype == np.float32:
            return (<DynamicArray2DCpp[float]*>self.thisptr).rows()
        elif self.dtype == np.int32:
            return (<DynamicArray2DCpp[int32_t]*>self.thisptr).rows()
        elif self.dtype == np.int64:
            return (<DynamicArray2DCpp[int64_t]*>self.thisptr).rows()
        elif self.dtype == np.bool_:
            return (<DynamicArray2DCpp[char]*>self.thisptr).rows()
        return 0

    cdef size_t get_cols(self):
        """C-level access to cols"""
        if self.dtype == np.float64:
            return (<DynamicArray2DCpp[double]*>self.thisptr).cols()
        elif self.dtype == np.float32:
            return (<DynamicArray2DCpp[float]*>self.thisptr).cols()
        elif self.dtype == np.int32:
            return (<DynamicArray2DCpp[int32_t]*>self.thisptr).cols()
        elif self.dtype == np.int64:
            return (<DynamicArray2DCpp[int64_t]*>self.thisptr).cols()
        elif self.dtype == np.bool_:
            return (<DynamicArray2DCpp[char]*>self.thisptr).cols()
        return 0


    cdef size_t get_stride(self):
        """C-level access to stride"""
        if self.dtype == np.float64:
            return (<DynamicArray2DCpp[double]*>self.thisptr).stride()
        elif self.dtype == np.float32:
            return (<DynamicArray2DCpp[float]*>self.thisptr).stride()
        elif self.dtype == np.int32:
            return (<DynamicArray2DCpp[int32_t]*>self.thisptr).stride()
        elif self.dtype == np.int64:
            return (<DynamicArray2DCpp[int64_t]*>self.thisptr).stride()
        elif self.dtype == np.bool_:
            return (<DynamicArray2DCpp[char]*>self.thisptr).stride()
        return 0

    def resize(self, tuple new_shape):
        """Resize array to new shape"""
        cdef size_t new_rows = new_shape[0]
        cdef size_t new_cols = new_shape[1]

        if self.dtype == np.float64:
            (<DynamicArray2DCpp[double]*>self.thisptr).resize(new_rows, new_cols)
        elif self.dtype == np.float32:
            (<DynamicArray2DCpp[float]*>self.thisptr).resize(new_rows, new_cols)
        elif self.dtype == np.int32:
            (<DynamicArray2DCpp[int32_t]*>self.thisptr).resize(new_rows, new_cols)
        elif self.dtype == np.int64:
            (<DynamicArray2DCpp[int64_t]*>self.thisptr).resize(new_rows, new_cols)
        elif self.dtype == np.bool_:
            (<DynamicArray2DCpp[char]*>self.thisptr).resize(new_rows, new_cols)

    def resize_along_first(self, new_shape):
        """Resize along first dimension (rows), keeping columns unchanged"""
        if isinstance(new_shape, int):
            new_rows = new_shape
        elif isinstance(new_shape, (tuple, list)):
            new_rows = new_shape[0]
        else:
            raise ValueError("new_shape must be int, tuple, or list")

        cdef size_t rows = new_rows
        if self.dtype == np.float64:
            (<DynamicArray2DCpp[double]*>self.thisptr).resize_along_first(rows)
        elif self.dtype == np.float32:
            (<DynamicArray2DCpp[float]*>self.thisptr).resize_along_first(rows)
        elif self.dtype == np.int32:
            (<DynamicArray2DCpp[int32_t]*>self.thisptr).resize_along_first(rows)
        elif self.dtype == np.int64:
            (<DynamicArray2DCpp[int64_t]*>self.thisptr).resize_along_first(rows)
        elif self.dtype == np.bool_:
            (<DynamicArray2DCpp[char]*>self.thisptr).resize_along_first(rows)

    def shrink(self, new_shape):
        """Shrink array to exact new shape, freeing unused memory"""
        cdef size_t new_rows
        cdef size_t new_cols
        if isinstance(new_shape, int):
            # Shrink just rows, keep cols
            new_rows = new_shape
            if self.dtype == np.float64:
                (<DynamicArray2DCpp[double]*>self.thisptr).shrink(new_rows)
            elif self.dtype == np.float32:
                (<DynamicArray2DCpp[float]*>self.thisptr).shrink(new_rows)
            elif self.dtype == np.int32:
                (<DynamicArray2DCpp[int32_t]*>self.thisptr).shrink(new_rows)
            elif self.dtype == np.int64:
                (<DynamicArray2DCpp[int64_t]*>self.thisptr).shrink(new_rows)
            elif self.dtype == np.bool_:
                (<DynamicArray2DCpp[char]*>self.thisptr).shrink(new_rows)
        else:
            # Shrink both dimensions
            new_rows = new_shape[0]
            new_cols = new_shape[1]
            if self.dtype == np.float64:
                (<DynamicArray2DCpp[double]*>self.thisptr).shrink(new_rows, new_cols)
            elif self.dtype == np.float32:
                (<DynamicArray2DCpp[float]*>self.thisptr).shrink(new_rows, new_cols)
            elif self.dtype == np.int32:
                (<DynamicArray2DCpp[int32_t]*>self.thisptr).shrink(new_rows, new_cols)
            elif self.dtype == np.int64:
                (<DynamicArray2DCpp[int64_t]*>self.thisptr).shrink(new_rows, new_cols)
            elif self.dtype == np.bool_:
                (<DynamicArray2DCpp[char]*>self.thisptr).shrink(new_rows, new_cols)

    @property
    def data(self):
        """
        The magic getter! This creates a zero-copy NumPy 'view' of our C++ data.
        It's not a copy; it's a direct window into the C++ memory, which is why it's so fast.
        Every time our code accesses `my_array.data`, this code runs to build that view on the fly.
        """
        # First, what's the logical shape the user sees,we get it ...
        cdef size_t rows = self.get_rows()
        cdef size_t cols = self.get_cols()
        # Now, the two most important pieces for our zero-copy trick:
        # 1. The actual memory address where our data lives in C++.
        cdef void* data_ptr = self.get_data_ptr()
        # 2. The *physical* width of a row in memory. This might be wider than `cols`
        #    if we've over-allocated space to make future growth faster.
        cdef size_t stride = self.get_stride()
        # How many bytes does one element take up? (e.g., 8 for a float64)
        cdef size_t itemsize = self.dtype.itemsize

        # Handle the boring edge case: if the array is empty, just give back an empty NumPy array.
        if rows == 0 or cols == 0:
            return np.array([], dtype=self.dtype).reshape((0, 0))

        # --- Now we create the "map" that tells NumPy how to navigate our C++ memory correctly ---

        # These are C-style arrays to hold the shape and the "stride map".
        cdef cnp.npy_intp shape[2]
        cdef cnp.npy_intp strides[2]

        # So the shape is easy as it's just the logical dimensions.
        shape[0] = rows
        shape[1] = cols

        # Now, the stride map. This tells NumPy how many *bytes* to jump to move through the data.
        # To move to the next item in the same row (j -> j+1), just jump by one item's size.
        strides[1] = itemsize
        # To move to the *next row* (i -> i+1), we have to jump over a whole physical row in memory.
        strides[0] = stride * itemsize

        # We also need to describe our data type (e.g., float64) to NumPy in its native C language.
        cdef cnp.PyArray_Descr* descr = PyArray_DescrFromType(self.numpy_type)

        # Now we set the permissions and properties for our numpy view
        # Let's start with a crucial permission: making the array writeable!
        # Without this, NumPy would make it read-only, and `arr[i] = x` would fail.
        cdef int flags = cnp.NPY_ARRAY_WRITEABLE

        # A little optimization: if the memory is perfectly packed (no extra space in rows),
        # we can tell NumPy it's "C-contiguous". This can speed up some operations.
        if stride == cols:
            flags |= cnp.NPY_ARRAY_C_CONTIGUOUS

        # Here we call the master C-API function, we give it:
        # the memory pointer, the shape map, the stride map, the data type, and the permissions.
        cdef cnp.ndarray result = <cnp.ndarray>PyArray_NewFromDescr(
            <PyTypeObject*>np.ndarray,
            descr,
            2,
            shape,
            strides,
            data_ptr,
            flags,  # Use our flags variable
            None
        )

        # By default, NumPy assumes it owns the data and will try to free it later.
        # But *our* C++ vector owns it! Clearing this flag prevents a double-free, which would crash the program.
        cnp.PyArray_CLEARFLAGS(result, cnp.NPY_ARRAY_OWNDATA)
        return result

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
        return DynamicArray1DClass(shape[0], dtype, factor)
    elif len(shape) == 2:
        return DynamicArray2DClass(shape, dtype, factor)
    else:
        raise ValueError(
            f"DynamicArray only supports 1D or 2D shapes. Got shape={shape} (dim={len(shape)})"
        )

def DynamicArray1D(shape, dtype=float, factor=2, use_numpy_resize=False, refcheck=True):
    """Create 1D dynamic array"""
    if isinstance(shape, int):
        shape = (shape,)
    return DynamicArray1DClass(shape[0], dtype, factor)
