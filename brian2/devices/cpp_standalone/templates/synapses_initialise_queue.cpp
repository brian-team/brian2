{% macro cpp_file() %}
#include "code_objects/{{codeobj_name}}.h"
{% if owner is defined %}
{% set pathobj = owner.name %}
{% endif %}
void _run_{{codeobj_name}}(double t) {
    double* real_delays = &({{pathobj}}.delay[0]);
    int* sources = &({{pathobj}}.sources[0]);
    const unsigned int n_synapses = {{pathobj}}.sources.size();
    {{pathobj}}.queue->prepare(real_delays, sources, n_synapses,
                               {{pathobj}}.dt);
}
{% endmacro %}

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

void _run_{{codeobj_name}}(double t);

#endif
{% endmacro %}
