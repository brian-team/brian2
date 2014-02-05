#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>
#include "spikequeue.h"

template<class scalar> class Synapses;
template<class scalar> class SynapticPathway;

template <class scalar>
class SynapticPathway
{
public:
	int Nsource, Ntarget;
	std::vector<scalar>& delay;
	std::vector<int> &sources;
	scalar dt;
	CSpikeQueue<scalar> *queue;
	SynapticPathway(int _Nsource, int _Ntarget, std::vector<scalar>& _delay, std::vector<int> &_sources,
					scalar _dt, int _spikes_start, int _spikes_stop)
		: Nsource(_Nsource), Ntarget(_Ntarget), delay(_delay), sources(_sources), dt(_dt)
	{
		this->queue = new CSpikeQueue<scalar>(_spikes_start, _spikes_stop);
	};
	~SynapticPathway()
	{
		if(this->queue) delete this->queue;
		this->queue = 0;
	}
};

template <class scalar>
class Synapses
{
public:
    int _N_value;
    inline double _N() { return _N_value;};
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
		_N_value = 0;
	};
};

#endif

