{# USES_VARIABLES { _count, _source_start, _source_stop, _spikespace, _num_source_neurons, t, _array_default_clock_t } #}

// Record spikes
{{support_code_lines}}

// For each spiking neuron
for(int _idx=0; _idx<_num_spikes; _idx++)
{
    const int _neuron_idx = {{_spikespace}}[_idx];
    {{vector_code|autoindent}}
}
