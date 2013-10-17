////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cpp_file() %}

// USES_VARIABLES { _synaptic_pre, _synaptic_post, _post_synaptic,
//                  _pre_synaptic, rand}

#include "{{codeobj_name}}.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"
#include "brianlib/synapses.h"

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

{% if variables is defined %}
{% set synpre = '_dynamic'+variables['_synaptic_pre'].arrayname %}
{% set synpost = '_dynamic'+variables['_synaptic_post'].arrayname %}
{% set synobj = owner.name %}
{% endif %}

// {{synobj}}

void _run_{{codeobj_name}}(double t)
{
	///// CONSTANTS ///////////
	%CONSTANTS%
	///// POINTERS ////////////
	{% for line in pointers_lines %}
	{{line}}
	{% endfor %}

	int _synapse_idx = {{synpre}}.size();
	for(int i=0; i<_num_all_pre; i++)
	{
		for(int j=0; j<_num_all_post; j++)
		{
		    const int _vectorisation_idx = j;
			// Define the condition
			{% for line in code_lines %}
			{{line}}
			{% endfor %}
			// Add to buffer
			if(_cond)
			{
			    if (_p != 1.0) {
			        // We have to use _rand instead of rand to use our rand
			        // function, not the one from the C standard library
			        if (_rand(_vectorisation_idx) >= _p)
			            continue;
			    }

			    for (int _repetition=0; _repetition<_n; _repetition++) {
			    	{{synpre}}.push_back(_pre_idcs);
			    	{{synpost}}.push_back(_post_idcs);
			    	{{synobj}}._pre_synaptic[i].push_back(_synapse_idx);
			    	{{synobj}}._post_synaptic[j].push_back(_synapse_idx);
                    _synapse_idx++;
                }
			}
		}
	}

	// now we need to resize all registered variables
	{% if owner is defined %}
	const int newsize = _dynamic{{owner.variables['_synaptic_pre'].arrayname}}.size();
	{% for variable in owner._registered_variables %}
	_dynamic{{variable.arrayname}}.resize(newsize);
	{% endfor %}
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

#endif
{% endmacro %}
