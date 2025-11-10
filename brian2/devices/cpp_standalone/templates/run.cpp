{% macro cpp_file() %}
#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include<random>

{% for codeobj in code_objects | sort(attribute='name') %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

{% for name in user_headers | sort %}
#include {{name}}
{% endfor %}

void brian_start()
{
	_init_arrays();
	_load_arrays();
	// Initialize clocks (link timestep and dt to the respective arrays)
    {% for clock in clocks | sort(attribute='name') %}
    brian::{{clock.name}}.timestep = brian::{{array_specs[clock.variables['timestep']]}};
    brian::{{clock.name}}.t = brian::{{array_specs[clock.variables['t']]}};
    {% if clock.__class__.__name__ == "EventClock" %}  {# FIXME: A bit ugly... #}
    brian::{{clock.name}}.times = brian::{{array_specs[clock.variables['times']]}};
    brian::{{clock.name}}.n_times = {{clock.variables['times'].size}};
    {% else %}
    brian::{{clock.name}}.dt = brian::{{array_specs[clock.variables['dt']]}};
    {% endif %}
    {% endfor %}
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}

{% for name, lines in run_funcs.items() | sort(attribute='name') %}
void {{name}}()
{
	using namespace brian;

    {{lines|autoindent}}
}

{% endfor %}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

void brian_start();
void brian_end();

{% for name, lines in run_funcs.items() | sort(attribute='name') %}
void {{name}}();
{% endfor %}

{% endmacro %}
