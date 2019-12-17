{# USES_VARIABLES { _n_sources, _n_targets, _source_dt } #}
{% macro cpp_file() %}
#include "code_objects/{{codeobj_name}}.h"
{% set pathobj = owner.name %}
void _run_{{codeobj_name}}() {
	using namespace brian;
    {{pointers_lines|autoindent}}
    {% set scalar = c_data_type(owner.variables['delay'].dtype) %}
    std::vector<{{scalar}}> &real_delays = {{get_array_name(owner.variables['delay'], access_data=False)}};
    {{scalar}}* real_delays_data = real_delays.empty() ? 0 : &(real_delays[0]);
    int32_t* sources = {{pathobj}}.sources.empty() ? 0 : &({{pathobj}}.sources[0]);
    const size_t n_delays = real_delays.size();
    const size_t n_synapses = {{pathobj}}.sources.size();
    {{pathobj}}.prepare({{constant_or_scalar('_n_sources', variables['_n_sources'])}},
                        {{constant_or_scalar('_n_targets', variables['_n_targets'])}},
                        real_delays_data, n_delays, sources,
                        n_synapses,
                        {{_source_dt}});
}
{% endmacro %}

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "objects.h"

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
