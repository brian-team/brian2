////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cpp_file() %}

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
	///// POINTERS ////////////
	{% for line in pointers_lines %}
	{{line}}
	{% endfor %}

	//// MAIN CODE ////////////
	for(int _idx=0; _idx<_num_idx; _idx++)
	{
		// THIS MESSAGE IS JUST TO LET YOU KNOW WE'RE IN THE STANDALONE NOT WEAVE TEMPLATE
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

void _run_{{codeobj_name}}(double t);

#endif
{% endmacro %}
