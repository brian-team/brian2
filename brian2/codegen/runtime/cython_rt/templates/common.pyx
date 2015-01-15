#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as _numpy
cimport numpy as _numpy
from libc.math cimport sin, cos, tan, sinh, cosh, tanh, exp, log, log10, sqrt, abs, asin, acos, atan, fabs, fmod, floor, ceil
cdef extern from "math.h":
    double M_PI
from libcpp cimport bool
from libc.stdint cimport int32_t, int64_t, uint8_t

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
