{# IS_OPENMP_COMPATIBLE #}
{% macro cpp_file() %}

#include "network.h"
#include<stdlib.h>
#include<stdio.h>
#include <time.h>
{{ openmp_pragma('include') }}

#define Clock_epsilon 1e-14

Network_construct(Network *network)
{
    network->first_obj = (void*)0;
    network->clock = 0;
	network->t = 0.0;
}

void Network_add(Network* network, Clock* clock, codeobj_func func)
{
    network->clock = clock;
    obj* new_obj = malloc(sizeof(obj));
    new_obj->func = func;
    new_obj->next = (void*)0;

	if (!network->first_obj)  // very first object
	{
	    network->first_obj = new_obj;
	} else {
		obj* current_obj = network->first_obj;
    	while (current_obj->next)
    	    current_obj = current_obj->next;
	    current_obj->next = new_obj;
    }
}

void Network_run(Network* network, const double duration, void (*report_func)(const double, const double, const double), const double report_period)
{
    const double t_start = network->t;
	const double t_end = network->t + duration;
	Clock_set_interval(network->clock, network->t, t_end);
	{{ openmp_pragma('parallel') }}
	{
		while(Clock_running(network->clock))
		{
	        obj* current_obj = network->first_obj;
			while (current_obj)
			{
	            current_obj->func();
                current_obj = current_obj->next;
			}
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

typedef struct {
  codeobj_func func;
  struct obj *next;
} obj;

typedef struct
{
	double t;
	Clock *clock;
	obj *first_obj;
} Network;

void Network_run(Network* network, const double duration, void (*report_func)(const double, const double, const double), const double report_period);
void Network_clear(Network* network);

#endif

{% endmacro %}
