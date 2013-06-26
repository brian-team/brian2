# USE_SPECIFIERS { _spikes }
_neuron_idx = _spikes
{% for line in code_lines %}
{{line}}
{% endfor %}
