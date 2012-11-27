Code examples for Synapses
--------------------------

A typical code is:
pre='v+=w'

This means: for every spike arriving at synapse s, do:
s.v+=s.w

Here, w is a synaptic variable but v is understood as the postsynaptic variable
v. So in object-oriented notation, this would mean:

(s.post)->v+=s.w

Now in Synapses, synaptic variables are stored in flat arrays, indexed by
a single synaptic index. So the above code would write:

for n in spiking_synapses:
	v[postsynaptic[n]]+=w[n]

Here n is the index of the synapse, spiking_synapses is an array of synapses
receiving spikes (given by the SpikeQueue), v is an array indexed on postsynaptic
neurons (NeuronGroup variable), postsynaptic and w are arrays indexed on
synapses. If we make it fully explicit:

for n in spiking_synapses:
	target_group.v[S.postsynaptic[n]]+=S.w[n]

This is much simpler than in Connection because there is no two-dimensional
indexing.
