{% extends 'common_group.cpp' %}
{# USES_VARIABLES { N } #}

{% block maincode %}
//// MAIN CODE ////////////
// scalar code
const size_t _vectorisation_idx = -1;
{# Note that the scalar_code['statement'] will not write to any scalar
   variables (except if the condition is simply 'True' and no vector code
   is present), it will only read in scalar variables that are used by the
   vector code. #}
{{scalar_code['condition']|autoindent}}
{{scalar_code['statement']|autoindent}}

{# N is a constant in most cases (NeuronGroup, etc.), but a scalar array for
   synapses, we therefore have to take care to get its value in the right
   way. #}
const int _N = {{constant_or_scalar('N', variables['N'])}};

{{ openmp_pragma('parallel-static') }}
for(int _idx=0; _idx<_N; _idx++)
{
    // vector code
    const size_t _vectorisation_idx = _idx;
    {{vector_code['condition']|autoindent}}
    if (_cond)
    {
        {{vector_code['statement']|autoindent}}
    }
}
{% endblock %}
