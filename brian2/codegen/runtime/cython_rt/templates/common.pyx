#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as _numpy
cimport numpy as _numpy
from libc.math cimport sin, cos, tan, sinh, cosh, tanh, exp, log, log10, sqrt, abs, asin, acos, atan, fabs, fmod
from libcpp cimport bool

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
