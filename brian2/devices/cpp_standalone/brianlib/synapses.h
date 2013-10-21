#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>

using namespace std;

template<class scalar> class Synapses;
template<class scalar> class SynapticPathway;

template <class scalar>
class SpikeQueue
{
public:
	SynapticPathway<scalar> &path;
	vector< vector<int> > queue; // queue[(offset+i)%queue.size()] is delay i relative to current time
	scalar dt;
	int offset;

	SpikeQueue(SynapticPathway<scalar> &_path, scalar _dt)
		: path(_path), dt(_dt)
	{
		queue.resize(1);
		offset = 0;
	};

	void expand(int newsize)
	{
		int n = queue.size();
		if(newsize<=n) return;
		// rotate offset back to start (leaves the circular structure unchanged)
		rotate(queue.begin(), queue.begin()+offset, queue.end());
		offset = 0;
		// add new elements
		queue.resize(newsize);
	};

	inline void ensure_delay(int delay)
	{
		if(delay>=queue.size())
		{
			expand(delay+1);
		}
	};

	void push(int *spikes, int nspikes)
	{
		// TODO: handle offsets for subgroups
		for(int idx_spike=0; idx_spike<nspikes; idx_spike++)
		{
			int idx_neuron = spikes[idx_spike];
			vector<int> &cur_indices = path.indices[idx_neuron];
			for(int idx_indices=0; idx_indices<cur_indices.size(); idx_indices++)
			{
				int synaptic_index = cur_indices[idx_indices];
				int delay = (int)(path.delay[synaptic_index]/dt+0.5); // rounds to nearest int
				// make sure there is enough space and resize if not
				ensure_delay(delay);
				// insert the index into the correct queue
				queue[(offset+delay)%queue.size()].push_back(synaptic_index);
			}
		}
	};

	inline vector<int>& peek()
	{
		return queue[offset];
	};

	void next()
	{
		// empty the current queue, note that for most compilers this shouldn't deallocate the memory,
		// although VC<=7.1 will, so it will be less efficient with that compiler
		queue[offset].clear();
		// and advance to the next offset
		offset = (offset+1)%queue.size();
	};
};

template <class scalar>
class SynapticPathway
{
public:
	int Nsource, Ntarget;
	vector<scalar>& delay;
	vector< vector<int> > &indices;
	scalar dt;
	SpikeQueue<scalar> *queue;
	SynapticPathway(int _Nsource, int _Ntarget, vector<scalar>& _delay, vector< vector<int> > &_indices,
					scalar _dt)
		: Nsource(_Nsource), Ntarget(_Ntarget), delay(_delay), indices(_indices), dt(_dt)
	{
		this->queue = new SpikeQueue<scalar>(*this, dt);
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

