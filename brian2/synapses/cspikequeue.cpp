#include<iostream>
#include<vector>
#include<map>
#include<iterator>
#include<algorithm>
#include"stdint_compat.h"
#include<assert.h>
using namespace std;

//TODO: The data type for indices is currently fixed (int), all floating point
//      variables (delays, dt) are assumed to use the same data type

template <class scalar>
class CSpikeQueue
{
public:
	vector< vector<int32_t> > queue; // queue[(offset+i)%queue.size()] is delay i relative to current time
	scalar dt;
	unsigned int offset;
	bool scalar_delay;
	unsigned int *delays;
	int32_t source_start;
	int32_t source_end;
    unsigned int openmp_padding;
    vector< vector<int> > synapses;
    // data structures for the store/restore mechanism
    map<string, vector< vector<int32_t> > > _stored_queue;
    map<string, unsigned int> _stored_offset;

    CSpikeQueue(int _source_start, int _source_end)
        : source_start(_source_start), source_end(_source_end)
    {
        queue.resize(1);
        offset = 0;
        dt = 0.0;
        delays = NULL;
        openmp_padding = 0;
        scalar_delay = 0;
    };

    void prepare(scalar *real_delays, unsigned int n_delays,
                 int32_t *sources, unsigned int n_synapses,
                 double _dt)
    {

        assert(n_delays == 1 || n_delays == n_synapses);
        scalar_delay = n_delays == 1;

        if (delays)
            delete [] delays;

        if (dt != 0.0 && dt != _dt)
        {
            // dt changed, we have to get the old spikes out of the queue and
            // reinsert them at the correct positions
            vector< vector<int32_t> > queue_copy = queue; // does a real copy
            const double conversion_factor = dt / _dt;
            const size_t oldsize = queue.size();
            const size_t newsize = (int)(oldsize * conversion_factor) + 1;
            queue.clear();
            queue.resize(newsize);
            for (unsigned int i=0; i<oldsize; i++)
            {
                vector<int32_t> spikes = queue_copy[(i + offset) % oldsize];
                queue[(int)(i * conversion_factor + 0.5)] = spikes;
            }
            offset = 0;
        }

        delays = new unsigned int[n_delays];
        synapses.clear();
        synapses.resize(source_end - source_start);

        for (unsigned int i=0; i<n_synapses; i++)
        {
            if (i == 0 || !scalar_delay)
            {
                //round to nearest int
                delays[i] =  (int)(real_delays[i] / _dt + 0.5);
            }
            synapses[sources[i] - source_start].push_back(i + openmp_padding);
        }

        dt = _dt;
    }

    void store(const string name)
    {
        _stored_queue[name].clear();
        _stored_queue[name].resize(queue.size());
        for (int i=0; i<queue.size(); i++)
            _stored_queue[name][i] = queue[i];
        _stored_offset[name] = offset;
    }

    void restore(const string name)
    {
        size_t size = _stored_queue[name].size();
        queue.clear();
        if (size == 0)  // the queue did not exist at the time of the store call
            size = 1;
        queue.resize(size);
        for (int i=0; i<_stored_queue[name].size(); i++)
            queue[i] = _stored_queue[name][i];
        offset = _stored_offset[name];
    }

    void expand(unsigned int newsize)
    {
        const unsigned int n = queue.size();
        if (newsize<=n)
            return;
        // rotate offset back to start (leaves the circular structure unchanged)
        rotate(queue.begin(), queue.begin()+offset, queue.end());
        offset = 0;
        // add new elements
        queue.resize(newsize);
    };

    inline void ensure_delay(unsigned int delay)
    {
        if(delay>=queue.size())
        {
            expand(delay+1);
        }
    };

	void push(int32_t *spikes, unsigned int nspikes)
	{
		const unsigned int start = static_cast<unsigned int>(distance(spikes, lower_bound(spikes, spikes+nspikes, source_start)));
		const unsigned int stop = static_cast<unsigned int>(distance(spikes, upper_bound(spikes, spikes+nspikes, source_end-1))); 
		for(unsigned int idx_spike=start; idx_spike<stop; idx_spike++)
		{
			const unsigned int idx_neuron = spikes[idx_spike] - source_start;
			vector<int> &cur_indices = synapses[idx_neuron];
			for(unsigned int idx_indices=0; idx_indices<cur_indices.size(); idx_indices++)
			{
				const int synaptic_index = cur_indices[idx_indices];
				unsigned int delay = scalar_delay ? delays[0] : delays[synaptic_index - openmp_padding];
				// make sure there is enough space and resize if not
				ensure_delay(delay);
				// insert the index into the correct queue
				queue[(offset+delay)%queue.size()].push_back(synaptic_index);
			}
		}
	};

	inline vector<int32_t>* peek()
	{
		return &queue[offset];
	};

    void advance()
    {
        // empty the current queue, note that for most compilers this shouldn't deallocate the memory,
        // although VC<=7.1 will, so it will be less efficient with that compiler
        queue[offset].clear();
        // and advance to the next offset
        offset = (offset+1)%queue.size();
    };
};
