#include<vector>
#include<algorithm>

using namespace std;

class SpikeQueue
{
public:
	vector< vector<int> > queue; // queue[(offset+i)%queue.size()] is delay i relative to current time
	double dt;
	int offset;

	SpikeQueue(double _dt)
		: dt(_dt)
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

	void push(int *spikes, double *delays, int nspikes)
	{
		int start = lower_bound(spikes, spikes+nspikes, path.spikes_start)-spikes;
		int stop = upper_bound(spikes, spikes+nspikes, path.spikes_stop)-spikes;
		for(int idx_spike=start; idx_spike<stop; idx_spike++)
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
