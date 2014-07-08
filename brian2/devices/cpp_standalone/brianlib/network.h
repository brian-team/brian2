#ifndef _BRIAN_NETWORK_H
#define _BRIAN_NETWORK_H

#include<vector>
#include<utility>
#include<set>
#include "clocks.h"

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
