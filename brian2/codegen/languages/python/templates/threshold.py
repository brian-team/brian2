# USE_SPECIFIERS { _num_neurons, refractory, refractory_until, t }
import numpy as np

_vectorisation_idx = np.arange(_num_neurons)
{% for line in code_lines %}
{{line}}
{% endfor %}
_return_values, = _cond.nonzero()
refractory_until[_return_values] = t + refractory[_return_values] 
