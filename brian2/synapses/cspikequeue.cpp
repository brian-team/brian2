#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;

class CSpikeQueue
{
public:
	vector< vector<int> > queue; // queue[(offset+i)%queue.size()] is delay i relative to current time
	double dt = 0.0;
	int offset;
	int *delays = NULL;
	int source_start;
	int source_end;
    vector< vector<int> > synapses;

	CSpikeQueue(int _source_start, int _source_end)
		: source_start(_source_start), source_end(_source_end)
	{
		queue.resize(1);
		offset = 0;
	};

    void prepare(double *real_delays, int *sources, int n_sources,
                 int n_synapses, double _dt)
    {
        if (delays)
            delete [] delays;

        delays = new int[n_synapses];
        synapses.clear();
        synapses.resize(n_sources);

        for (int i=0; i<n_synapses; i++)
        {
            delays[i] =  (int)(real_delays[i] / _dt + 0.5); //round to nearest int
            synapses[sources[i] - source_start].push_back(i);
        }

        dt = _dt;
    }

	void expand(int newsize)
	{
		const int n = queue.size();
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
		const int start = lower_bound(spikes, spikes+nspikes, source_start)-spikes;
		const int stop = upper_bound(spikes, spikes+nspikes, source_end)-spikes;
		for(int idx_spike=start; idx_spike<stop; idx_spike++)
		{
			const int idx_neuron = spikes[idx_spike] - source_start;
			vector<int> &cur_indices = synapses[idx_neuron];
			for(int idx_indices=0; idx_indices<cur_indices.size(); idx_indices++)
			{
				const int synaptic_index = cur_indices[idx_indices];
				const int delay = delays[synaptic_index];
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
