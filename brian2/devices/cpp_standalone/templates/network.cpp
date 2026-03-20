{% macro cpp_file() %}

#include "network.h"
#include<stdlib.h>
#include<iostream>
#include <ctime>
#include<utility>
#include<chrono>

{{ openmp_pragma('include') }}

#define Clock_epsilon 1e-14

double Network::_last_run_time = 0.0;
double Network::_last_run_completed_fraction = 0.0;
bool Network::_globally_stopped = false;
bool Network::_globally_running = false;

Network::Network()
{
    t = 0.0;
}

void Network::clear()
{
    objects.clear();
}

void Network::add(BaseClock* clock, codeobj_func func)
{
#if defined(_MSC_VER) && (_MSC_VER>=1700)
    objects.push_back(std::make_pair(std::move(clock), std::move(func)));
#else
    objects.push_back(std::make_pair(clock, func));
#endif
}

void Network::run(const double duration, void (*report_func)(const double, const double, const double, const double), const double report_period)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, current;
    const double t_start = t;
    const double t_end = t + duration;
    double next_report_time = report_period;
    // compute the set of clocks
    compute_clocks();
    // set interval for all clocks

    for(std::set<BaseClock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
        (*i)->set_interval(t, t_end);

    start = std::chrono::high_resolution_clock::now();
    if (report_func)
        report_func(0.0, 0.0, t_start, duration);

    BaseClock* clock = next_clocks();
    double elapsed_realtime = 0.0;
    bool did_break_early = false;

    {% if maximum_run_time is not none %}
    const bool should_check_time = true;
    {% else %}
    const bool should_check_time = (report_func != NULL);
    {% endif %}

    Network::_globally_running = true;
    Network::_globally_stopped = false;
    while(clock && clock->running() && !Network::_globally_stopped)
    {
        t = clock->t[0];

        if (should_check_time)
        {
            current = std::chrono::high_resolution_clock::now();
            elapsed_realtime = std::chrono::duration<double>(current - start).count();

            {% if maximum_run_time is not none %}
            if (elapsed_realtime > {{maximum_run_time}})
            {
                did_break_early = true;
                break;
            }
            {% endif %}

            if (report_func && elapsed_realtime > next_report_time)
            {
                report_func(elapsed_realtime, (t - t_start)/duration, t_start, duration);
                next_report_time += report_period;
            }
        }

        for(size_t i=0; i<objects.size(); i++)
        {
            BaseClock *obj_clock = objects[i].first;
            if (curclocks.find(obj_clock) != curclocks.end())
            {
                codeobj_func func = objects[i].second;
                if (func)  // code objects can be NULL in cases where we store just the clock
                    func();
            }
        }
        for(std::set<BaseClock*>::iterator i=curclocks.begin(); i!=curclocks.end(); i++)
            (*i)->tick();
        clock = next_clocks();
    }
    Network::_globally_running = false;
    current = std::chrono::high_resolution_clock::now();
    elapsed_realtime = std::chrono::duration<double>(current - start).count();

    if(!did_break_early && !Network::_globally_stopped)
        t = t_end;
    else
        t = clock->t[0];

    _last_run_time = elapsed_realtime;
    if(duration > 0)
        _last_run_completed_fraction = (t - t_start) / duration;
    else
        _last_run_completed_fraction = 1.0;

    if (report_func)
        report_func(elapsed_realtime, _last_run_completed_fraction, t_start, duration);
}

void Network::compute_clocks()
{
    clocks.clear();
    for(int i=0; i<objects.size(); i++)
    {
        BaseClock *clock = objects[i].first;
        clocks.insert(clock);
    }
}

BaseClock* Network::next_clocks()
{
    if (clocks.empty())
        return NULL;
    // find minclock, clock with smallest t value
    BaseClock *minclock = *clocks.begin();

    for(std::set<BaseClock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
    {
        BaseClock *clock = *i;
        if(clock->t[0]<minclock->t[0])
            minclock = clock;
    }
    // find set of equal clocks
    curclocks.clear();

    double t = minclock->t[0];
    for(std::set<BaseClock*>::iterator i=clocks.begin(); i!=clocks.end(); i++)
    {
        BaseClock *clock = *i;
        double s = clock->t[0];
        if(s==t || fabs(s-t)<=Clock_epsilon)
            curclocks.insert(clock);
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
    std::set<BaseClock*> clocks, curclocks;
    void compute_clocks();
    BaseClock* next_clocks();
public:
    std::vector< std::pair< BaseClock*, codeobj_func > > objects;
    double t;
    static double _last_run_time;
    static double _last_run_completed_fraction;
    static bool _globally_stopped;
    static bool _globally_running;

    Network();
    void clear();
    void add(BaseClock *clock, codeobj_func func);
    void run(const double duration, void (*report_func)(const double, const double, const double, const double), const double report_period);
};

#endif

{% endmacro %}
