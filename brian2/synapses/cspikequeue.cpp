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
    int offset;
    bool scalar_delay;
    int *delays;
    int32_t source_start;
    int32_t source_end;
    int openmp_padding;
    vector< vector<int> > synapses;
    // data structures for the store/restore mechanism

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

    ~CSpikeQueue()
    {
        if (delays)
        {
            delete[] delays;
            delays = NULL;
        }
    }

    void prepare(scalar *real_delays, int n_delays,
                 int32_t *sources, int n_synapses,
                 double _dt)
    {

        assert(n_delays == 1 || n_delays == n_synapses);

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
            for (size_t i=0; i<oldsize; i++)
            {
                vector<int32_t> spikes = queue_copy[(i + offset) % oldsize];
                queue[(int)(i * conversion_factor + 0.5)] = spikes;
            }
            offset = 0;
        }

        delays = new int[n_delays];
        synapses.clear();
        synapses.resize(source_end - source_start);

        // Note that n_synapses and n_delays do not have to be identical
        // (homogeneous delays are stored as a single scalar), we therefore
        // use two independent loops to initialize the delays and the synapses
        // array
        scalar first_delay = n_delays > 0 ? real_delays[0] : 0.0;
        int min_delay = (int)(first_delay / _dt + 0.5);
        int max_delay = min_delay;
        for (int i=0; i<n_delays; i++)
        {
            //round to nearest int
            delays[i] =  (int)(real_delays[i] / _dt + 0.5);
            if (delays[i] > max_delay)
                max_delay = delays[i];
            else if (delays[i] < min_delay)
                min_delay = delays[i];
        }
        for (int i=0; i<n_synapses; i++)
            synapses[sources[i] - source_start].push_back(i + openmp_padding);

        dt = _dt;

        // Ensure that our spike queue is sufficiently big
        ensure_delay(max_delay);

        scalar_delay = (min_delay == max_delay);
    }

    pair <int, vector< vector<int32_t> > > _full_state()
    {
        pair <int, vector< vector<int32_t> > > state(offset, queue);
        return state;
    }

    void _clear()
    {
    }

    void _restore_from_full_state(const pair <int, vector< vector<int32_t> > > state)
    {
        int stored_offset = state.first;
        vector< vector<int32_t> > stored_queue = state.second;
        size_t size = stored_queue.size();
        queue.clear();
        if (size == 0)  // the queue did not exist at the time of the store call
            size = 1;
        queue.resize(size);
        for (size_t i=0; i<stored_queue.size(); i++)
            queue[i] = stored_queue[i];
        offset = stored_offset;
    }

    void expand(size_t newsize)
    {
        const size_t n = queue.size();
        if (newsize <= n)
            return;
        // rotate offset back to start (leaves the circular structure unchanged)
        rotate(queue.begin(), queue.begin()+offset, queue.end());
        offset = 0;
        // add new elements
        queue.resize(newsize);
    };

    inline void ensure_delay(int delay)
    {
        if(delay >= (int)queue.size())
        {
            expand(delay+1);
        }
    };

    void push(int32_t *spikes, int nspikes)
    {
        if(nspikes==0) return;
        const int start = static_cast<int>(distance(spikes, lower_bound(spikes, spikes+nspikes, source_start)));
        const int stop = static_cast<int>(distance(spikes, upper_bound(spikes, spikes+nspikes, source_end-1)));
        const int32_t * __restrict rspikes = spikes;
        if(scalar_delay)
        {
            vector<int32_t> &homog_queue = queue[(offset+delays[0])%queue.size()];
            for(int idx_spike=start; idx_spike<stop; idx_spike++)
            {
                const int idx_neuron = rspikes[idx_spike] - source_start;
                const int num_indices = synapses[idx_neuron].size();
                if(num_indices==0) continue;
                const int* __restrict cur_indices = &(synapses[idx_neuron][0]);
                const int cur_homog_queue_size = homog_queue.size();
                homog_queue.resize(cur_homog_queue_size+num_indices);
                int32_t * __restrict hq = &(homog_queue[cur_homog_queue_size]);
                for(int idx_indices=0; idx_indices<num_indices; idx_indices++)
                {
                    hq[idx_indices] = cur_indices[idx_indices];
                }
            }
        } else // (!scalar_delay)
        {
            int * __restrict rdelays = delays-openmp_padding;
            for(int idx_spike=start; idx_spike<stop; idx_spike++)
            {
                const int idx_neuron = rspikes[idx_spike] - source_start;
                const int num_indices = synapses[idx_neuron].size();
                if(num_indices==0) continue;
                const int* __restrict cur_indices = &(synapses[idx_neuron][0]);
                for(int idx_indices=0; idx_indices<num_indices; idx_indices++)
                {
                    const int synaptic_index = cur_indices[idx_indices];
                    int delay = rdelays[synaptic_index];
                    queue[(offset+delay)%queue.size()].push_back(synaptic_index);
                }
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
