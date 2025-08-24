{# USES_VARIABLES { t, _indices, N } #}

{{support_code_lines}}

// Update time array
// ...

// Record state variables
for(int _i=0; _i<_num_indices; _i++)
{
    const int _idx = {{_indices}}[_i];
    const int _vectorisation_idx = _idx;

    {{vector_code|autoindent}}
}
