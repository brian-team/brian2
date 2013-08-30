////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cpp_file() %}
// USES_VARIABLES { not_refractory, lastspike, t, _spikespace }

#include "{{codeobj_name}}.h"
#include<math.h>
#include<stdint.h>

////// SUPPORT CODE ///////
{% for line in support_code_lines %}
{{line}}
{% endfor %}

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
	long _cpp_numspikes = 0;
	for(int _idx=0; _idx<_num_idx; _idx++)
	{
	    const int _vectorisation_idx = _idx;
		{% for line in code_lines %}
		{{line}}
		{% endfor %}
		if(_cond) {
			_spikespace[_cpp_numspikes++] = _idx;
			// We have to use the pointer names directly here: The condition
			// might contain references to not_refractory or lastspike and in
			// that case the names will refer to a single entry.
			_ptr{{_array_not_refractory}}[_idx] = false;
			_ptr{{_array_lastspike}}[_idx] = t;
		}
	}
	_spikespace[_num_idx] = _cpp_numspikes;
}
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// HEADER FILE ///////////////////////////////////////////////////////////

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "arrays.h"

void _run_{{codeobj_name}}(double t);

#endif
{% endmacro %}
