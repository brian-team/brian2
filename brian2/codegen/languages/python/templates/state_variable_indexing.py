# USE_SPECIFIERS { _num_neurons }
import numpy as np
_vectorisation_idx = np.arange(_num_neurons)
{% for line in code_lines %}
{{line}}
{% endfor %}
_return_values, = _cond.nonzero()
