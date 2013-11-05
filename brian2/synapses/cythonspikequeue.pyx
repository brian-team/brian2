# distutils: language = c++
# distutils: sources = brian2/synapses/cspikequeue.cpp
from libcpp.vector cimport vector

cimport numpy as np

cdef extern from "cspikequeue.cpp":
    cdef cppclass CSpikeQueue:
        CSpikeQueue(int, int) except +
        void prepare(double*, int*, int, int, double)
        void push(int *, int)
        vector[int] peek()
        void next()

ctypedef np.int32_t DTYPE_int
ctypedef np.float64_t DTYPE_float

cdef class SpikeQueue:
    cdef CSpikeQueue *thisptr

    def __cinit__(self, int source_start, int source_end):
        self.thisptr = new CSpikeQueue(source_start, source_end)

    def __dealloc__(self):
        del self.thisptr

    def prepare(self, np.ndarray[DTYPE_float, ndim=1] real_delays not None,
                double dt,
                np.ndarray[DTYPE_int, ndim=1] sources not None):
        self.thisptr.prepare(<double*>real_delays.data,
                             <int*>sources.data, sources.shape[0],
                             real_delays.shape[0], dt)

    def push(self, np.ndarray[DTYPE_int, ndim=1] spikes not None):
        self.thisptr.push(<int*>spikes.data, spikes.shape[0])

    def peek(self):
        return self.thisptr.peek()

    def next(self):
        self.thisptr.next()
