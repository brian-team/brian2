////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cpp_file() %}

#include "{{codeobj_name}}.h"
#include<math.h>
#include<stdint.h>

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
{% set _spikespace = variables['_spikespace'].arrayname %}
{% set _i = '_dynamic'+variables['_i'].arrayname %}
{% set _t = '_dynamic'+variables['_t'].arrayname %}
{% endif %}

void _run_{{codeobj_name}}(double t)
{
	///// CONSTANTS ///////////
	%CONSTANTS%
	///// POINTERS ////////////
	{% for line in pointers_lines %}
	{{line}}
	{% endfor %}

	//// MAIN CODE ////////////
	int _num_spikes = {{_spikespace}}[_num_{{_spikespace}}-1];
    if (_num_spikes > 0)
    {
        int _start_idx = 0;
        int _end_idx = - 1;
        for(int _i=0; _i<_num_spikes; _i++)
        {
            const int _idx = {{_spikespace}}[_i];
            if (_idx >= _source_start) {
                _start_idx = _i;
                break;
            }
        }
        for(int _i=_start_idx; _i<_num_spikes; _i++)
        {
            const int _idx = {{_spikespace}}[_i];
            if (_idx >= _source_end) {
                _end_idx = _i;
                break;
            }
        }
        if (_end_idx == -1)
            _end_idx =_num_spikes;
        _num_spikes = _end_idx - _start_idx;
        if (_num_spikes > 0) {
        	for(int _i=_start_idx; _i<_end_idx; _i++)
        	{
        		const int _idx = {{_spikespace}}[_i];
        		{{_i}}.push_back(_idx-_source_start);
        		{{_t}}.push_back(t);
        	}
        }
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
