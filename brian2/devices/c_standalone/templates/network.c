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
    const double t_start = {{net.name}}->t;
	const double t_end = {{net.name}}->t + duration;
	Clock_set_interval({{net.name}}->clock, {{net.name}}->t, t_end);
	{{ openmp_pragma('parallel') }}
	{
		while(Clock_running({{net.name}}->clock))
		{
		    // Simulate all the objects for this time step
		    {% for codeobj in code_objects %}
	        _run_{{codeobj.name}}();
	        {% endfor %}
			Clock_tick({{net.name}}->clock);
		}
	{{net.name}}->t = t_end;
	}
}


{% endmacro %}

{% macro h_file() %}

#ifndef _BRIAN_NETWORK_H
#define _BRIAN_NETWORK_H

#include "brianlib/clocks.h"

typedef void (*codeobj_func)();

typedef struct
{
	double t;
	codeobj_func* objects;
	int n_objects;
	Clock *clock;
} Network;

void Network_run_{{net.name}}(const double duration);
void Network_clear(Network* network);

#endif

{% endmacro %}
