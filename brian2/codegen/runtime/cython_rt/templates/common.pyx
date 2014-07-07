#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as _numpy
cimport numpy as _numpy
from libc.math cimport sin, cos, tan, sinh, cosh, tanh, exp, log, log10, sqrt, abs, asin, acos, atan, fabs, fmod

# support code
{{ support_code | autoindent }}

def main(_namespace):
    cdef int _idx
    cdef int _vectorisation_idx
    {{ load_namespace | autoindent }}
    {% block maincode %}
    {{ vector_code | autoindent }}
    {% endblock %}
