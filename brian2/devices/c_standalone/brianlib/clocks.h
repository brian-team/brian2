#ifndef _BRIAN_CLOCKS_H
#define _BRIAN_CLOCKS_H

#include<math.h>

typedef struct
{
	int i, i_end;
	double epsilon;
    double dt_value;
} Clock;

extern void Clock_construct(Clock *clock, double _dt, double _epsilon);

extern void Clock_reinit(Clock *clock);
extern void Clock_tick(Clock *clock);
extern double Clock_dt_(Clock *clock);
extern double Clock_t_(Clock *clock);
extern double Clock_t_end(Clock *clock);
extern char Clock_running(Clock *clock);
extern void Clock_set_interval(Clock *clock, double start, double end);

#endif
