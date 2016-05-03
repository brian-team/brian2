#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True

import numpy as _numpy
from numba import jit

# support code
{{ support_code | autoindent }}

# template-specific support code
{% block template_support_code %}
{% endblock %}

@jit
def main(_namespace):
    {{ load_namespace | autoindent }}
    if '_owner' in _namespace:
        _owner = _namespace['_owner']
    {% block maincode %}
    {{ vector_code | autoindent }}
    {% endblock %}
