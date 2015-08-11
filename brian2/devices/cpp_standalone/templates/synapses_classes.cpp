{# IS_OPENMP_COMPATIBLE #}
{% macro cpp_file() %}
{% endmacro %}

{% macro h_file() %}

#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>
{{ openmp_pragma('include') }}

#include "brianlib/spikequeue.h"

template<class scalar> class Synapses;
template<class scalar> class SynapticPathway;

template <class scalar>
class SynapticPathway
{
public:
	int Nsource, Ntarget, _nb_threads;
	std::vector<scalar> &delay;
	std::vector<int> &sources;
	std::vector<int> all_peek;
	scalar dt;
	std::vector< CSpikeQueue<scalar> * > queue;
	SynapticPathway(int _Nsource, int _Ntarget, std::vector<scalar>& _delay, std::vector<int> &_sources,
					scalar _dt, int _spikes_start, int _spikes_stop)
		: Nsource(_Nsource), Ntarget(_Ntarget), delay(_delay), sources(_sources), dt(_dt)
	{
	   _nb_threads = {{ openmp_pragma('get_num_threads') }};

	   for (int _idx=0; _idx < _nb_threads; _idx++)
	       queue.push_back(new CSpikeQueue<scalar>(_spikes_start, _spikes_stop));
    };

	~SynapticPathway()
	{
		for (int _idx=0; _idx < _nb_threads; _idx++)
			delete(queue[_idx]);
	}

	void push(int *spikes, unsigned int nspikes)
    {
    	queue[{{ openmp_pragma('get_thread_num') }}]->push(spikes, nspikes);
    }

	void advance()
    {
    	queue[{{ openmp_pragma('get_thread_num') }}]->advance();
    }

	vector<int32_t>* peek()
    {
    	{{ openmp_pragma('static-ordered') }}
		for(int _thread=0; _thread < {{ openmp_pragma('get_num_threads') }}; _thread++)
		{
			{{ openmp_pragma('ordered') }}
			{
    			if (_thread == 0)
					all_peek.clear();
				all_peek.insert(all_peek.end(), queue[_thread]->peek()->begin(), queue[_thread]->peek()->end());
    		}
    	}
   
    	return &all_peek;
    }

    void prepare(scalar *real_delays, unsigned int n_delays,
                 int *sources, unsigned int n_synapses, double _dt)
    {
    	{{ openmp_pragma('parallel') }}
    	{
            unsigned int length;
            if ({{ openmp_pragma('get_thread_num') }} == _nb_threads - 1) 
                length = n_synapses - (unsigned int) {{ openmp_pragma('get_thread_num') }}*(n_synapses/_nb_threads);
            else
                length = (unsigned int) n_synapses/_nb_threads;
    		
            unsigned int padding  = {{ openmp_pragma('get_thread_num') }}*(n_synapses/_nb_threads);

            queue[{{ openmp_pragma('get_thread_num') }}]->openmp_padding = padding;
            if (n_delays > 1)
    		    queue[{{ openmp_pragma('get_thread_num') }}]->prepare(&real_delays[padding], length, &sources[padding], length, _dt);
    		else
    		    queue[{{ openmp_pragma('get_thread_num') }}]->prepare(&real_delays[0], 1, &sources[padding], length, _dt);
    	}
    }

};

template <class scalar>
class Synapses
{
public:
	int Nsource;
	int Ntarget;
	std::vector< std::vector<int> > _pre_synaptic;
	std::vector< std::vector<int> > _post_synaptic;

	Synapses(int _Nsource, int _Ntarget)
		: Nsource(_Nsource), Ntarget(_Ntarget)
	{
		for(int i=0; i<Nsource; i++)
			_pre_synaptic.push_back(std::vector<int>());
		for(int i=0; i<Ntarget; i++)
			_post_synaptic.push_back(std::vector<int>());
	};
};

#endif

{% endmacro %}