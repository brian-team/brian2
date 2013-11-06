# distutils: language = c++
# distutils: sources = brian2/synapses/cspikequeue.cpp

from libcpp.vector cimport vector

cimport numpy as np
import numpy as np

np.import_array()

ctypedef np.int32_t DTYPE_int
ctypedef np.float64_t DTYPE_float

cdef extern from "cspikequeue.cpp":
    cdef cppclass CSpikeQueue:
        CSpikeQueue(int, int) except +
        void prepare(double*, int*, int, int, double)
        void push(int *, int)
        vector[DTYPE_int] peek()
        void next()

cdef class SpikeQueue:
    cdef CSpikeQueue *thisptr

    def __cinit__(self, int source_start, int source_end):
        self.thisptr = new CSpikeQueue(source_start, source_end)

    def __dealloc__(self):
        del self.thisptr

    def prepare(self, np.ndarray[DTYPE_float, ndim=1, mode='c'] real_delays,
                double dt,
                np.ndarray[DTYPE_int, ndim=1, mode='c'] sources):
        self.thisptr.prepare(<double*>real_delays.data,
                             <int*>sources.data, sources.shape[0],
                             real_delays.shape[0], dt)

    def push(self, np.ndarray[DTYPE_int, ndim=1, mode='c'] spikes):
        self.thisptr.push(<int*>spikes.data, spikes.shape[0])

    def peek(self):
        # This should create a numpy array from a std::vector<int> without
        # copying -- &spikes[0] is guaranteed to point to a contiguous array
        # according to the C++ standard.
        cdef:
            vector[DTYPE_int] spikes = self.thisptr.peek()
            DTYPE_int* spikes_data = &(spikes[0])
            unsigned int spikes_size = spikes.size()

        if spikes_size == 0:
            return np.empty(0, dtype=np.int)

        cdef np.npy_intp shape[1]
        cdef DTYPE_int[:] spikes_data_view = <DTYPE_int[:spikes_size]> spikes_data
        # FIXME: For some reason Cython changes the data type here from int32
        #        to int64 (therefore it also has to copy the data...?), using
        #        int32 here will lead to useless numbers in the array on the
        #        Python side
        cdef np.ndarray[np.int_t, ndim=1, mode='c'] ar = np.asarray(spikes_data_view,
                                                                    dtype=np.int,
                                                                    order='C')
        return <np.ndarray[np.int_t, ndim=1, mode='c']>ar

    def next(self):
        self.thisptr.next()
