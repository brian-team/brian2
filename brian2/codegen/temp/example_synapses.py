'''
TODO: support for synapses
In order to get this working, we need to have support for multiple indexes, and
for indexes to come from arrays. So for C++ we would get something like this:

for(int idx=0; idx<num_spikes_synapses; idx++)
{
        const int i = spike_synapses[idx];
        const int postsyn_idx = postsynaptic[i];
        double &v = _arr_v[postsyn_idx];
        const double w = _arr_w[i];
        v += w;
}
'''