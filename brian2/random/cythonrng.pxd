# Cython declaration file for the global random number generator
# Other Cython modules can use: from brian2.random.cythonrng cimport _rand, _randn

cdef double _rand() noexcept nogil
cdef double _randn() noexcept nogil
