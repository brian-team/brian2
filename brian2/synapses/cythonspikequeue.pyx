# distutils: language = c++
# distutils: sources = brian2/synapses/cspikequeue.cpp

from libcpp.vector cimport vector

from cython.operator import dereference
from cython.operator cimport dereference

cimport numpy as np
import numpy as np

np.import_array()

cdef extern from "inttypes.h":
    # Note that this does not actually define int32_t as int (which might be
    # wrong on a 64bit system), it only tells Cython that int32_t is an int-like
    # type
    ctypedef int int32_t

cdef extern from "cspikequeue.cpp":
    cdef cppclass CSpikeQueue[T]:
        CSpikeQueue(int, int) except +
        void prepare(T*, int32_t*, int, int, double)
        void push(int32_t *, int)
        vector[int32_t]* peek()
        void advance()

cdef class SpikeQueue:
    # TODO: Currently, the data type for dt and delays is fixed
    cdef CSpikeQueue[double] *thisptr

    def __cinit__(self, int source_start, int source_end):
        self.thisptr = new CSpikeQueue[double](source_start, source_end)

    def __dealloc__(self):
        del self.thisptr

    def prepare(self, np.ndarray[double, ndim=1, mode='c'] real_delays,
                double dt,
                np.ndarray[int32_t, ndim=1, mode='c'] sources):
        self.thisptr.prepare(<double*>real_delays.data,
                             <int32_t*>sources.data, sources.shape[0],
                             real_delays.shape[0], dt)

    def push(self, np.ndarray[int32_t, ndim=1, mode='c'] spikes):
        self.thisptr.push(<int32_t*>spikes.data, spikes.shape[0])

    def peek(self):
        # This should create a numpy array from a std::vector<int> without
        # copying -- &spikes[0] is guaranteed to point to a contiguous array
        # according to the C++ standard.
        cdef:
            vector[int32_t]* spikes = self.thisptr.peek()
            int32_t* spikes_data = &(dereference(spikes)[0])
            unsigned int spikes_size = dereference(spikes).size()

        if spikes_size == 0:
            return np.empty(0, dtype=np.int32)

        cdef np.npy_intp shape[1]
        shape[0] = spikes_size
        return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, spikes_data)

    def advance(self):
        self.thisptr.advance()
