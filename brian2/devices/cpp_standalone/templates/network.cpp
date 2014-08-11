{% macro cpp_file() %}

#include "network.h"
#include<stdlib.h>
#include<iostream>
{{ openmp_pragma('include') }}

#define Clock_epsilon 1e-14

Network::Network()
{
	t = 0.0;
}

void Network::clear()
{
	objects.clear();
}

void Network::add(Clock* clock, codeobj_func func)
{
	objects.push_back(std::make_pair<Clock*, codeobj_func>(clock, func));
}

void Network::run(double duration)
{
	double t_end = t + duration;
	// compute the set of clocks
	compute_clocks();
	// set interval for all clocks

	{{ openmp_pragma('single') }}
	{
		for(std::set<Clock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
		{
			(*i)->set_interval(t, t_end);
		}
	}

	Clock* clock = next_clocks();
	
	{{ openmp_pragma('parallel') }}
	{
		while(clock->running())
		{
			for(int i=0; i<objects.size(); i++)
			{
				Clock *obj_clock = objects[i].first;
				// Only execute the object if it uses the right clock for this step
				if (curclocks.find(obj_clock) != curclocks.end())
				{
	                codeobj_func func = objects[i].second;
	                {{ openmp_pragma('barrier') }}
	                func();
				}
			}
			{{ openmp_pragma('single') }}
			{
				for(std::set<Clock*>::iterator i=curclocks.begin(); i!=curclocks.end(); i++)
				{
					(*i)->tick();
				}
			}
			clock = next_clocks();
		}
	t = t_end;
	}
}

void Network::compute_clocks()
{
	clocks.clear();
	for(int i=0; i<objects.size(); i++)
	{
		Clock *clock = objects[i].first;
		clocks.insert(clock);
	}
}

Clock* Network::next_clocks()
{
	// find minclock, clock with smallest t value
	Clock *minclock = *clocks.begin();
	for(std::set<Clock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
	{
		Clock *clock = *i;
		if(clock->t_()<minclock->t_())
			minclock = clock;
	}
	// find set of equal clocks
	{{ openmp_pragma('single') }}
	{
		// find set of equal clocks
		curclocks.clear();

		double t = minclock->t_();
		for(std::set<Clock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
		{
			Clock *clock = *i;
			double s = clock->t_();
			if(s==t or fabs(s-t)<=Clock_epsilon)
				curclocks.insert(clock);
		}
	}
	return minclock;
}

{% endmacro %}

{% macro h_file() %}

#ifndef _BRIAN_NETWORK_H
#define _BRIAN_NETWORK_H

#include<vector>
#include<utility>
#include<set>
#include "brianlib/clocks.h"

typedef void (*codeobj_func)();

class Network
{
	std::set<Clock*> clocks, curclocks;
	void compute_clocks();
	Clock* next_clocks();
public:
	std::vector< std::pair< Clock*, codeobj_func > > objects;
	double t;

	Network();
	void clear();
	void add(Clock *clock, codeobj_func func);
	void run(double duration);
};

#endif

{% endmacro %}