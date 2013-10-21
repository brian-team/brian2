////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cpp_file() %}

#include "code_objects/{{codeobj_name}}.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"
#include<iostream>

////// SUPPORT CODE ///////
namespace {
	{% for line in support_code_lines %}
	{{line}}
	{% endfor %}
}

////// HASH DEFINES ///////
{% for line in hashdefine_lines %}
{{line}}
{% endfor %}

void _run_{{codeobj_name}}(double t)
{
	///// CONSTANTS ///////////
	%CONSTANTS%
	///// POINTERS ////////////
	{% for line in pointers_lines %}
	{{line}}
	{% endfor %}

	//// MAIN CODE ////////////
	{% if pathway is defined %}
	vector<int> &_spiking_synapses = {{pathway.name}}.queue->peek();
	const int _num_spiking_synapses = _spiking_synapses.size();
	{% endif %}
	for(int _spiking_synapse_idx=0;
		_spiking_synapse_idx<_num_spiking_synapses;
		_spiking_synapse_idx++)
	{
		const int _idx = _spiking_synapses[_spiking_synapse_idx];
		const int _vectorisation_idx = _idx;
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
	}
}

void _debugmsg_{{codeobj_name}}()
{
	{% if owner is defined %}
	cout << "Number of synapses: " << _dynamic_array_{{owner.name}}__synaptic_pre.size() << endl;
	{% endif %}
}

{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// HEADER FILE ///////////////////////////////////////////////////////////

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "objects.h"

void _run_{{codeobj_name}}(double t);
void _debugmsg_{{codeobj_name}}();

#endif
{% endmacro %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
