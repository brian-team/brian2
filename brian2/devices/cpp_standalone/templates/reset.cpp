////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cpp_file() %}
	// USES_VARIABLES { _spikes }

#include "{{codeobj_name}}.h"
#include<math.h>

////// SUPPORT CODE ///////
{% for line in support_code_lines %}
{{line}}
{% endfor %}

////// HASH DEFINES ///////
{% for line in hashdefine_lines %}
{{line}}
{% endfor %}

void _run_{{codeobj_name}}()
{
	///// POINTERS ////////////
	{% for line in pointers_lines %}
	{{line}}
	{% endfor %}

	//// MAIN CODE ////////////
	for(int _index_spikes=0; _index_spikes<_num_spikes; _index_spikes++)
	{
		const int _idx = _spikes[_index_spikes];
		const int _vectorisation_idx = _idx;
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
	}
}
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// HEADER FILE ///////////////////////////////////////////////////////////

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "arrays.h"

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
