{# USES_VARIABLES { _n_sources, _n_targets, _source_dt } #}
{% macro cpp_file() %}
#include "code_objects/{{codeobj_name}}.h"
{% set pathobj = owner.name %}
void _run_{{codeobj_name}}() {
	using namespace brian;
	{{pointers_lines|autoindent}}

    double* real_delays = {{pathobj}}.delay.empty() ? 0 : &({{pathobj}}.delay[0]);
    int32_t* sources = {{pathobj}}.sources.empty() ? 0 : &({{pathobj}}.sources[0]);
    const unsigned int n_delays = {{pathobj}}.delay.size();
    const unsigned int n_synapses = {{pathobj}}.sources.size();
    {{pathobj}}.prepare({{constant_or_scalar('_n_sources', variables['_n_sources'])}},
                        {{constant_or_scalar('_n_targets', variables['_n_targets'])}},
                        real_delays, n_delays, sources,
                        n_synapses,
                        {{_source_dt}});
}
{% endmacro %}

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
