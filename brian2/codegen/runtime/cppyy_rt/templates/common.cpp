{# Base template for groups - handles most common cases #}
{# Don't declare specific variables here - let child templates do it #}

// Support code
{{support_code_lines}}

// Scalar code
{% if scalar_code %}
{
    const int _vectorisation_idx = -1;
    {{scalar_code|autoindent}}
}
{% endif %}

// Vector code
{% if vector_code %}
{
    // Get N if available
    {% if 'N' in variables %}
    const int _N = {{constant_or_scalar('N', variables['N'])}};
    {% else %}
    const int _N = 1;  // Default
    {% endif %}

    // Main loop
    for(int _idx=0; _idx<_N; _idx++)
    {
        const int _vectorisation_idx = _idx;
        {{vector_code|autoindent}}
    }
}
{% endif %}
