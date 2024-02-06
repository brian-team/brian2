{% macro cython_directives() %}
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
#cython: cpow=True
#cython: infer_types=True
{% endmacro %}

{% macro imports() %}
import numpy as _numpy
cimport numpy as _numpy
from libc.math cimport fabs, sin, cos, tan, sinh, cosh, tanh, exp, log, log10, expm1, log1p, sqrt, asin, acos, atan, fmod, floor, ceil, isinf
cdef extern from "math.h":
    double M_PI
# Import the two versions of std::abs
from libc.stdlib cimport abs  # For integers
from libc.math cimport abs  # For floating point values
from libc.limits cimport INT_MIN, INT_MAX
from libcpp cimport bool
from libcpp.set cimport set
from cython.operator cimport dereference as _deref, preincrement as _preinc
cimport cython as _cython

_numpy.import_array()
cdef extern from "numpy/ndarraytypes.h":
    void PyArray_CLEARFLAGS(_numpy.PyArrayObject *arr, int flags)
from libc.stdlib cimport free

cdef extern from "numpy/npy_math.h":
    bint npy_isinf(double x)
    double NPY_INFINITY

cdef extern from "stdint_compat.h":
    # Longness only used for type promotion
    # Actual compile time size used for conversion
    ctypedef signed int int32_t
    ctypedef signed long int64_t
    ctypedef unsigned long uint64_t
    # It seems we cannot used a fused type here
    cdef int int_(bool)
    cdef int int_(char)
    cdef int int_(short)
    cdef int int_(int)
    cdef int int_(long)
    cdef int int_(float)
    cdef int int_(double)
    cdef int int_(long double)
{% endmacro %}

{% macro before_run() %}
{{ cython_directives() }}
{{ imports() }}

# support code
{{ support_code_lines | autoindent }}

def main(_namespace):
    {{ load_namespace | autoindent }}
    if '_owner' in _namespace:
        _owner = _namespace['_owner']
    {% block before_code %}
    # EMPTY_CODE_BLOCK  -- overwrite in child template
    {% endblock %}
{% endmacro %}

{% macro run() %}
{{ cython_directives() }}
{{ imports() }}

# support code
{{ support_code_lines | autoindent }}

# template-specific support code
{% block template_support_code %}
{% endblock %}

def main(_namespace):
    cdef size_t _idx
    cdef size_t _vectorisation_idx
    {{ load_namespace | autoindent }}
    if '_owner' in _namespace:
        _owner = _namespace['_owner']
    {% block maincode %}
    {% endblock %}

{% endmacro %}

{% macro after_run() %}
{{ cython_directives() }}
{{ imports() }}

# support code
{{support_code_lines | autoindent}}

def main(_namespace):
    {{ load_namespace | autoindent }}
    if '_owner' in _namespace:
        _owner = _namespace['_owner']
    {% block after_code %}
    # EMPTY_CODE_BLOCK  -- overwrite in child template
    {% endblock %}
{% endmacro %}
