////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cpp_file() %}

#include "code_objects/{{codeobj_name}}.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

void _run_{{codeobj_name}}(double t)
{
	{% if owner is defined %}
	// we do advance at the beginning rather than at the end because it saves us making
	// a copy of the current spiking synapses
	{{owner.name}}.queue->advance();
	{{owner.name}}.queue->push({{owner.source.variables['_spikespace'].arrayname}}, {{owner.source.variables['_spikespace'].arrayname}}[{{owner.source|length}}]);
	{{owner.name}}.queue->peek();
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
