{% macro cpp_file() %}

#include<stdint.h>
#include<vector>
#include "objects.h"
#include "brianlib/synapses.h"

// static arrays
{% for (varname, dtype_spec, N) in array_specs %}
{{dtype_spec}} *{{varname}};
const int _num_{{varname}} = {{N}};
{% endfor %}

// dynamic arrays
{% for (varname, dtype_spec) in dynamic_array_specs %}
std::vector<{{dtype_spec}}> {{varname}};
{% endfor %}

// synapses
{% for S in synapses %}
Synapses<double> _synapses_{{S.name}}({{S.source|length}}, {{S.target|length}});
// temporarily hardcoded pathway and queue
SynapticPathway<double> _synaptic_pathway_{{S.name}}_pre({{S.source|length}}, {{S.target|length}},
		_dynamic_array_{{S.name}}_pre_delay, _synapses_{{S.name}}._pre_synaptic);
SpikeQueue<double> _spike_queue_{{S.name}}_pre(_synaptic_pathway_{{S.name}}_pre, {{S.source.clock.dt_}});
{% endfor %}

void _init_arrays()
{
    // Arrays initialized to 0
	{% for (varname, dtype_spec, N) in zero_specs %}
	{{varname}} = new {{dtype_spec}}[{{N}}];
	for(int i=0; i<{{N}}; i++) {{varname}}[i] = 0;
	{% endfor %}

	// Arrays initialized to an "arange"
	{% for (varname, dtype_spec, start, stop) in arange_specs %}
	{{varname}} = new {{dtype_spec}}[{{stop}}-{{start}}];
	for(int i=0; i<{{stop}}-{{start}}; i++) {{varname}}[i] = {{start}} + i;
	{% endfor %}
}

void _dealloc_arrays()
{
	{% for (varname, dtype_spec, N) in array_specs %}
	if({{varname}}!=0)
	{
		delete [] {{varname}};
		{{varname}} = 0;
	}
	{% endfor %}
}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<vector>
#include<stdint.h>
#include "brianlib/synapses.h"

// static arrays
{% for (varname, dtype_spec, N) in array_specs %}
extern {{dtype_spec}} *{{varname}};
extern const int _num_{{varname}};
{% endfor %}

// dynamic arrays
{% for (varname, dtype_spec) in dynamic_array_specs %}
extern std::vector<{{dtype_spec}}> {{varname}};
{% endfor %}

// synapses
{% for S in synapses %}
extern Synapses<double> _synapses_{{S.name}};
extern SynapticPathway<double> _synaptic_pathway_{{S.name}}_pre;
extern SpikeQueue<double> _spike_queue_{{S.name}}_pre;
{% endfor %}

void _init_arrays();
void _dealloc_arrays();

#endif

{% endmacro %}
