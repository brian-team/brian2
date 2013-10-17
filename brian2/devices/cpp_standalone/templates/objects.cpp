{% macro cpp_file() %}

#include<stdint.h>
#include<vector>
#include "objects.h"
#include "brianlib/synapses.h"

//////////////// static arrays ////////////
{% for (varname, dtype_spec, N) in array_specs %}
{{dtype_spec}} *{{varname}};
const int _num_{{varname}} = {{N}};
{% endfor %}

//////////////// dynamic arrays ///////////
{% for (varname, dtype_spec) in dynamic_array_specs %}
std::vector<{{dtype_spec}}> {{varname}};
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses %}
// {{S.name}}
Synapses<double> {{S.name}}({{S.source|length}}, {{S.target|length}});
{% for path in S._pathways %}
SynapticPathway<double> {{path.name}}(
		{{S.source|length}}, {{S.target|length}},
		_dynamic{{path.variables['delay'].arrayname}},
		{{S.name}}._{{path.prepost}}_synaptic,
		{{S.source.dt_}}
		);
{% endfor %}
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

//////////////// static arrays ////////////
{% for (varname, dtype_spec, N) in array_specs %}
extern {{dtype_spec}} *{{varname}};
extern const int _num_{{varname}};
{% endfor %}

//////////////// dynamic arrays ///////////
{% for (varname, dtype_spec) in dynamic_array_specs %}
extern std::vector<{{dtype_spec}}> {{varname}};
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses %}
// {{S.name}}
extern Synapses<double> {{S.name}};
{% for path in S._pathways %}
extern SynapticPathway<double> {{path.name}};
{% endfor %}
{% endfor %}

void _init_arrays();
void _dealloc_arrays();

#endif

{% endmacro %}
