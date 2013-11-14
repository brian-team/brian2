#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>
#include "spikequeue.h"

using namespace std;

template<class scalar> class Synapses;
template<class scalar> class SynapticPathway;

template <class scalar>
class SynapticPathway
{
public:
	int Nsource, Ntarget;
	vector<scalar>& delay;
	vector<int> &sources;
	scalar dt;
	CSpikeQueue<scalar> *queue;
	SynapticPathway(int _Nsource, int _Ntarget, vector<scalar>& _delay, vector<int> &_sources,
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
    int _N;
	int Nsource;
	int Ntarget;
	vector< vector<int> > _pre_synaptic;
	vector< vector<int> > _post_synaptic;

	Synapses(int _Nsource, int _Ntarget)
		: Nsource(_Nsource), Ntarget(_Ntarget)
	{
		for(int i=0; i<Nsource; i++)
			_pre_synaptic.push_back(vector<int>());
		for(int i=0; i<Ntarget; i++)
			_post_synaptic.push_back(vector<int>());
		_N = 0;
	};
};

#endif

