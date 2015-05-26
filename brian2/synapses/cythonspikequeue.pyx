# distutils: language = c++
# distutils: sources = brian2/synapses/cspikequeue.cpp

from libcpp.vector cimport vector
from libcpp.string cimport string

import cython
from cython.operator import dereference
from cython.operator cimport dereference

cimport numpy as np
import numpy as np

np.import_array()

cdef extern from "stdint_compat.h":
    # Longness only used for type promotion
    # Actual compile time size used for conversion
    ctypedef signed int int32_t
    ctypedef signed long int64_t

cdef extern from "cspikequeue.cpp":
    cdef cppclass CSpikeQueue[T]:
        CSpikeQueue(int, int) except +
        void prepare(T*, int, int32_t*, unsigned int, double)
        void push(int32_t *, unsigned int)
        void store(const string)
        void restore(const string)
        vector[int32_t]* peek()
        void advance()

cdef class SpikeQueue:
    # TODO: Currently, the data type for dt and delays is fixed
    cdef CSpikeQueue[double] *thisptr

    def __cinit__(self, int source_start, int source_end):
        self.thisptr = new CSpikeQueue[double](source_start, source_end)

    def __dealloc__(self):
        del self.thisptr

    def _store(self, str name='default'):
        cdef string s = name.encode('UTF-8')
        self.thisptr.store(s)

    def _restore(self, str name='default'):
        cdef string s = name.encode('UTF-8')
        self.thisptr.restore(s)

    def prepare(self, np.ndarray[double, ndim=1, mode='c'] real_delays,
                double dt,
                np.ndarray[int32_t, ndim=1, mode='c'] sources):
        self.thisptr.prepare(<double*>real_delays.data,
                             real_delays.shape[0],
                             <int32_t*>sources.data,
                             sources.shape[0], dt)

    def push(self, np.ndarray[int32_t, ndim=1, mode='c'] spikes):
        self.thisptr.push(<int32_t*>spikes.data, spikes.shape[0])

    def peek(self):
        # will only be used if the queue has size > 0
        cdef int32_t* spikes_data
        cdef np.npy_intp shape[1]
        # This should create a numpy array from a std::vector<int> without
        # copying -- &spikes[0] is guaranteed to point to a contiguous array
        # according to the C++ standard.
        cdef vector[int32_t]* spikes = self.thisptr.peek()
        cdef size_t spikes_size = dereference(spikes).size()
        if spikes_size == 0:
            return np.empty(0, dtype=np.int32)
        else:
            spikes_data = <int32_t*>(&(dereference(spikes)[0]))
            shape[0] = spikes_size
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, spikes_data)

    def advance(self):
        self.thisptr.advance()
