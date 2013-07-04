# USE_SPECIFIERS { _indices, refractory, refractory_until, t }
_vectorisation_idx = _indices
{% for line in code_lines %}
{{line}}
{% endfor %}
_return_values, = _cond.nonzero()
refractory_until[_return_values] = t + refractory[_return_values] 
