# distutils: language = c++
# distutils: sources = brian2/synapses/cspikequeue.cpp

from libcpp.vector cimport vector

from cython.operator import dereference
from cython.operator cimport dereference

cimport numpy as np
import numpy as np

np.import_array()

ctypedef np.int32_t DTYPE_int
ctypedef np.float64_t DTYPE_float

cdef extern from "cspikequeue.cpp":
    cdef cppclass CSpikeQueue[T]:
        CSpikeQueue(int, int) except +
        void prepare(T*, int*, int, int, double)
        void push(int *, int)
        vector[DTYPE_int]& peek()
        void advance()

cdef class SpikeQueue:
    # TODO: Currently, the data type for dt and delays is fixed
    cdef CSpikeQueue[DTYPE_float] *thisptr

    def __cinit__(self, int source_start, int source_end):
        self.thisptr = new CSpikeQueue[DTYPE_float](source_start, source_end)

    def __dealloc__(self):
        del self.thisptr

    def prepare(self, np.ndarray[DTYPE_float, ndim=1, mode='c'] real_delays,
                DTYPE_float dt,
                np.ndarray[DTYPE_int, ndim=1, mode='c'] sources):
        self.thisptr.prepare(<DTYPE_float*>real_delays.data,
                             <int*>sources.data, sources.shape[0],
                             real_delays.shape[0], dt)

    def push(self, np.ndarray[DTYPE_int, ndim=1, mode='c'] spikes):
        self.thisptr.push(<int*>spikes.data, spikes.shape[0])

    def peek(self):
        # This should create a numpy array from a std::vector<int> without
        # copying -- &spikes[0] is guaranteed to point to a contiguous array
        # according to the C++ standard.
        cdef:
            vector[DTYPE_int]* spikes = <vector[DTYPE_int]*>&(self.thisptr.peek())
            DTYPE_int* spikes_data = &(dereference(spikes)[0])
            unsigned int spikes_size = dereference(spikes).size()

        if spikes_size == 0:
            return np.empty(0, dtype=np.int)

        cdef np.npy_intp shape[1]
        shape[0] = spikes_size
        return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, spikes_data)

    def advance(self):
        self.thisptr.advance()
