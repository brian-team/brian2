{% macro cpp_file() %}
#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include "randomkit.h"

{% for codeobj in code_objects | sort(attribute='name') %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

{% for name in user_headers | sort %}
#include {{name}}
{% endfor %}

void {{simname}}::_start()
{
	_init_arrays();
	_load_arrays();
	// Initialize clocks (link timestep and dt to the respective arrays)
    {% for clock in clocks | sort(attribute='name') %}
    {{clock.name}}.timestep = {{array_specs[clock.variables['timestep']]}};
    {{clock.name}}.dt = {{array_specs[clock.variables['dt']]}};
    {{clock.name}}.t = {{array_specs[clock.variables['t']]}};
    {% endfor %}
    for (int i=0; i<{{openmp_pragma('get_num_threads')}}; i++)
	    rk_randomseed(_mersenne_twister_states[i]);  // Note that this seed can be potentially replaced in main.cpp
}

void {{simname}}::_end()
{
	_write_arrays();
	_dealloc_arrays();
}

{% for name, lines in run_funcs.items() | sort(attribute='name') %}
void {{simname}}::_run_func_{{name}}()
{
    {{lines|autoindent}}
}

{% endfor %}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}
{% endmacro %}
