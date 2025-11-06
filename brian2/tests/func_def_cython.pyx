#cython: language_level=3
cdef double foo(double x, const double y):
    return x + y + 3
