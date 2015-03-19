{# IS_OPENMP_COMPATIBLE #}
{% macro cpp_file() %}

#include "network.h"
#include<stdlib.h>
#include<stdio.h>
#include <time.h>
{{ openmp_pragma('include') }}

#define Clock_epsilon 1e-14

void Network_run(Network* network, const double duration)
{
    const double t_start = network->t;
	const double t_end = network->t + duration;
	Clock_set_interval(network->clock, network->t, t_end);
	{{ openmp_pragma('parallel') }}
	{
		while(Clock_running(network->clock))
		{
		    // Simulate all the objects for this time step
		    for (int i=0; i<network->n_objects; i++)
	            network->objects[i]();
			Clock_tick(network->clock);
		}
	network->t = t_end;
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

void Network_run(Network* network, const double duration);
void Network_clear(Network* network);

#endif

{% endmacro %}
