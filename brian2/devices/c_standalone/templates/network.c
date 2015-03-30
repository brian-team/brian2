{# IS_OPENMP_COMPATIBLE #}
{% macro cpp_file() %}

#include "network.h"
#include "objects.h"
{% for codeobj in code_objects | sort(attribute='name') %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}
#include<stdlib.h>
#include<stdio.h>
#include <time.h>
{{ openmp_pragma('include') }}

#define Clock_epsilon 1e-14

void Network_run_{{net.name}}(const double duration)
{
    const double t_start = {{net.name}}_t;
	const double t_end = {{net.name}}_t + duration;
	Clock_set_interval({{clock.name}}, {{net.name}}_t, t_end);
	{{ openmp_pragma('parallel') }}
	{
		while(Clock_running({{clock.name}}))
		{
		    // Simulate all the objects for this time step
		    {% for codeobj in code_objects %}
	        _run_{{codeobj.name}}();
	        {% endfor %}
			Clock_tick({{clock.name}});
		}
	{{net.name}}_t = t_end;
	}
}


{% endmacro %}

{% macro h_file() %}

#ifndef _BRIAN_NETWORK_H
#define _BRIAN_NETWORK_H

#include "brianlib/clocks.h"

extern double {{net.name}}_t;
void Network_run_{{net.name}}(const double duration);

#endif

{% endmacro %}
