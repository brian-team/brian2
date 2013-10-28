#ifndef _BRIAN_CLOCKS_H
#define _BRIAN_CLOCKS_H

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
	double dt;
	int i, i_end;
	double epsilon;
	Clock(double _dt, double _epsilon=1e-14) : dt(_dt), epsilon(_epsilon) { i=0; i_end=0; };
	inline void reinit() { i = 0; };
	inline void tick() { i += 1; };
	inline double t() { return i*dt; };
	inline double t_end() { return i_end*dt; };
	inline bool running() { return i<i_end; };
	void set_interval(double start, double end)
	{
        int i_start = fround(start/dt);
        double t_start = i_start*dt;
        if(t_start==start or fabs(t_start-start)<=epsilon*fabs(t_start))
        {
            i = i_start;
        } else
        {
            i = (int)ceil(start/dt);
        }
        i_end = fround(end/dt);
        double t_end = i_end*dt;
        if(not (t_end==end or fabs(t_end-end)<=epsilon*fabs(t_end)))
        {
            i_end = (int)ceil(end/dt);
        }
	}
};

#endif
