#ifndef _BRIAN_CLOCKS_H
#define _BRIAN_CLOCKS_H
#include<stdlib.h>
#include<iostream>
#include<algorithm>
#include<brianlib/stdint_compat.h>
#include<math.h>

namespace {
	inline int64_t fround(double x)
	{
		return (int64_t)(x+0.5);
	};
};

class BaseClock
{
public:
	int64_t *timestep;
	double *t;
	virtual void tick() = 0;
	virtual void set_interval(double start, double end) = 0;
	inline bool running() { return timestep[0]<i_end; };
protected:
	int64_t i_end;
};

class Clock : public BaseClock
{
public:
	double epsilon;
	double *dt;
	Clock(double _epsilon=1e-14) : epsilon(_epsilon) { i_end = 0;};
    inline void tick()
    {
        timestep[0] += 1;
        t[0] = timestep[0] * dt[0];
    }
	void set_interval(double start, double end)
	{
        int64_t i_start = fround(start/dt[0]);
        double t_start = i_start*dt[0];
        if(t_start==start || fabs(t_start-start)<=epsilon*fabs(t_start))
        {
            timestep[0] = i_start;
        } else
        {
            timestep[0] = (int64_t)ceil(start/dt[0]);
        }
        i_end = fround(end/dt[0]);
        double t_end = i_end*dt[0];
        if(!(t_end==end || fabs(t_end-end)<=epsilon*fabs(t_end)))
        {
            i_end = (int64_t)ceil(end/dt[0]);
        }
	}
};

class EventClock : public BaseClock
{
public:
	double *times;
    size_t n_times;

	EventClock() { i_end = 0;};
    inline void tick()
    {
       timestep[0] += 1;
       t[0] = times[timestep[0]];
    }
	void set_interval(double start, double end)
	{
        timestep[0] = std::lower_bound(times, times + n_times, start) - times;
        t[0] = times[timestep[0]];
        i_end = std::lower_bound(times, times + n_times, end) - times;
	}
};

#endif
