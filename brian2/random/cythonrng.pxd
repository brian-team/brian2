# Cython declaration file for the global random number generator
# Other Cython modules can use: from brian2.random.cythonrng cimport _rand, _randn

 # Note we accept (and ignore) the _idx parameter for backwards compatibility
cdef double _rand(int _idx) noexcept nogil
cdef double _randn(int _idx) noexcept nogil
