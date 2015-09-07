#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
#cython: infer_types=True

import numpy as _numpy
cimport numpy as _numpy
from libc.math cimport sin, cos, tan, sinh, cosh, tanh, exp, log, log10, sqrt, abs, asin, acos, atan, fabs, fmod, floor, ceil
cdef extern from "math.h":
    double M_PI
from libcpp cimport bool

cdef extern from "stdint_compat.h":
    # Longness only used for type promotion
    # Actual compile time size used for conversion
    ctypedef signed int int32_t
    ctypedef signed long int64_t
    ctypedef unsigned long uint64_t

# support code
{{ support_code | autoindent }}

# template-specific support code
{% block template_support_code %}
{% endblock %}

def main(_namespace):
    cdef int _idx
    cdef int _vectorisation_idx
    {{ load_namespace | autoindent }}
    if '_owner' in _namespace:
        _owner = _namespace['_owner']
    {% block maincode %}
    {{ vector_code | autoindent }}
    {% endblock %}
