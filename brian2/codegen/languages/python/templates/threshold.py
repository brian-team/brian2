# USE_SPECIFIERS { refractory, refractory_until, t }

{% for line in code_lines %}
{{line}}
{% endfor %}
_return_values, = _cond.nonzero()
refractory_until[_return_values] = t + refractory[_return_values] 
