#include<stdlib.h>
#include<math.h>
#include"clocks.h"

int fround(double x)
{
    return (int)(x+0.5);
};


void Clock_construct(Clock *clock, double _dt, double _epsilon)
{
    clock->dt_value = _dt;
    clock->epsilon = _epsilon;
    clock->i = 0;
    clock->i_end = 0;
}

void Clock_reinit(Clock *clock)
{
    clock->i = 0;
}

void Clock_tick(Clock *clock)
{
    clock->i += 1;
};

double Clock_dt_(Clock *clock) {
    return clock->dt_value;
}

double Clock_t_(Clock *clock)
{
    return clock->i * clock->dt_value;
}

double Clock_t_end(Clock *clock)
{
    return clock->i_end * clock->dt_value;
}

char Clock_running(Clock *clock)
{
    return clock->i < clock->i_end;
}

void Clock_set_interval(Clock *clock, double start, double end)
{
    int i_start = fround(start/clock->dt_value);
    double t_start = i_start*clock->dt_value;
    if(t_start==start || fabs(t_start-start)<=clock->epsilon*fabs(t_start))
    {
        clock->i = i_start;
    } else
    {
        clock->i = (int)ceil(start/clock->dt_value);
    }
    clock->i_end = fround(end/clock->dt_value);
    double t_end = clock->i_end*clock->dt_value;
    if(!(t_end==end || fabs(t_end-end)<=clock->epsilon*fabs(t_end)))
    {
        clock->i_end = (int)ceil(end/clock->dt_value);
    }
}