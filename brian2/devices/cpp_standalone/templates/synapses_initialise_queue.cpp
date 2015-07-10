{# IS_OPENMP_COMPATIBLE #}
{% macro cpp_file() %}
#include "code_objects/{{codeobj_name}}.h"
{% set pathobj = owner.name %}
void _run_{{codeobj_name}}() {
	using namespace brian;
    double* real_delays = {{pathobj}}.delay.empty() ? 0 : &({{pathobj}}.delay[0]);
    int32_t* sources = {{pathobj}}.sources.empty() ? 0 : &({{pathobj}}.sources[0]);
    const unsigned int n_delays = {{pathobj}}.delay.size();
    const unsigned int n_synapses = {{pathobj}}.sources.size();
    {{pathobj}}.prepare(real_delays, n_delays, sources, n_synapses,
                        {{pathobj}}.dt);
}
{% endmacro %}

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
