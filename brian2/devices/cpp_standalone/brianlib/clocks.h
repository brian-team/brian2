#ifndef _BRIAN_CLOCKS_H
#define _BRIAN_CLOCKS_H
#include<stdlib.h>
#include<iostream>
#include<brianlib/stdint_compat.h>
#include<math.h>

namespace {
	inline int fround(double x)
	{
		return (int)(x+0.5);
	};
};

class Clock
{
public:
	double epsilon;
	double *dt;
	int64_t *timestep;
	double *t;
	Clock(double _epsilon=1e-14) : epsilon(_epsilon) { i_end = 0;};
    inline void tick()
    {
        timestep[0] += 1;
        t[0] = timestep[0] * dt[0];
    }
	inline bool running() { return timestep[0]<i_end; };
	void set_interval(double start, double end)
	{
        int i_start = fround(start/dt[0]);
        double t_start = i_start*dt[0];
        if(t_start==start || fabs(t_start-start)<=epsilon*fabs(t_start))
        {
            timestep[0] = i_start;
        } else
        {
            timestep[0] = (int)ceil(start/dt[0]);
        }
        i_end = fround(end/dt[0]);
        double t_end = i_end*dt[0];
        if(!(t_end==end || fabs(t_end-end)<=epsilon*fabs(t_end)))
        {
            i_end = (int)ceil(end/dt[0]);
        }
	}
private:
	int64_t i_end;
};

#endif
