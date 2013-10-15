#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>

using namespace std;

class Synapses
{
public:
	vector< vector<int> > _pre_synaptic;
	vector< vector<int> > _post_synaptic;
	void init(int Nsource, int Ntarget)
	{
		for(int i=0; i<Nsource; i++)
			_pre_synaptic.push_back(vector<int>());
		for(int i=0; i<Ntarget; i++)
			_post_synaptic.push_back(vector<int>());
	}
};

#endif

