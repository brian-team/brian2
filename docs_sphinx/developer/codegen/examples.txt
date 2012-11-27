Code generation examples
++++++++++++++++++++++++

This is a list of things that code generation should be able to do.

Examples
========

State update
------------

Starting from a series of statements defining a state updater::

	_tmp_V := -V/tau
	V += _tmp_V*dt
	
We should generate Python code::

	_tmp_V = -V/tau
	V += _tmp_V*dt
	
Note that if we had written V = V+_tmp_V*dt then we would need to generate::

	V[:] = V+_tmp_V*dt
	
C++ code::

	for(int i=0; i<N; i++)
	{
		double &V = _arr_V[i];
		const double _tmp_V = -V/tau;
		V += _tmp_V*dt;
	}
	
GPU code::

	__global__ stateupdate(double t, double dt)
	{
		const int i = threadIdx.x+blockDim.x*blockIdx.x;
		if(i>=N) return;
		__volatile__ double &V = _arr_V[i];
		const double _tmp_V = -V/tau;
		V += _tmp_V*dt;
	}
	
Reset
-----

Reset works much like state update but is only applied on a subset of neurons.
Therefore starting from something like::

	V = Vr
	
We should generate Python code::

	V[spikes] = Vr[spikes]
	
C++ code::

	for(int sp=0; sp<Nspikes; sp++)
	{
		const int i = spikes[sp];
		double &V = _arr_V[i];
		const double Vr = _arr_Vr[i];
		V = Vr;
	}
	
GPU code::

	__global__ reset(int Nspikes)
	{
		const int sp = threadIdx.x+blockDim.x*blockIdx.x;
		if(sp>=Nspikes) return;
		const int i = spikes[sp]; // coalesced
		__volatile__ double &V = _arr_V[i]; // potentially uncoalesced
		const double Vr = _arr_Vr[i]; // potentially uncoalesced
		V = Vr;
	}

Threshold
---------

Starting from an expression that evaluates to bool::

	V>Vt
	
Python code::

	cond = V>Vt
	return cond.nonzero()[0]
	
C++ code::

	int numspikes = 0;
	for(int i=0; i<N; i++)
	{
		const double V = _arr_V[i];
		const double Vt = _arr_Vt[i];
		const bool cond = V>Vt;
		if(cond)
		{
			spikes[numspikes++] = i;
		}
	}
	return numspikes;
	
In the case of C++ code, we would also need to find a mechanism to return the
number of spikes (not possible with weave I think) and we would need to
preallocate a spikes array for it.

For GPU code, it's not entirely clear the best way to do it. Probably the
easiest is to have an array cond of bools, and then have a separate
compaction function that isn't part of the code generation framework that
returns a list of spikes. This increases memory bandwidth requirements, so an
alternative would be to integrate a compaction scheme with the threshold
checking. This would be much more efficient but potentially tricky to do.
For the first case, we would want to generate GPU code like this::

	__global__ threshold()
	{
		const int i = threadIdx.x+blockDim.x*blockIdx.x;
		if(i>=N) return;
		const double V = _arr_V[i];
		const double Vt = _arr_Vt[i];
		_arr_cond[i] = V>Vt;	
	}

Synapses
--------

Starting from a presynaptic statement::

	v += w

Romain suggested that the general scheme is something like this::

	for n in spiking_synapses:
		v[postsynaptic[n]]+=w[n]
		
So for C++ code I propose::

	for(int idx=0; idx<num_spikes_synapses; idx++)
	{
		const int i = spike_synapses[idx];
		const int postsyn_idx = postsynaptic[i];
		double &v = _arr_v[postsyn_idx];
		const double w = _arr_w[i];
		v += w;
	}

There will also be some complications to handle which variables are pre- or
post-synaptic in the case of, e.g., v being both pre and post, but this will
need to be handled before it gets to code generation, and that information
made available to code generation somehow. Presumably there is also a
presynaptic[i]?
	
Python code is slightly more tricky because we cannot handle it in one go, but
instead we use the trick developed by Victor and me::

    _post_neurons = _post.data.take(_synapses)
    _perm = _post_neurons.argsort()
    _aux = _post_neurons.take(_perm)
    _flag = empty(len(_aux)+1, dtype=bool)
    _flag[0] = _flag[-1] = 1
    not_equal(_aux[1:], _aux[:-1], _flag[1:-1])
    _F = _flag.nonzero()[0][:-1]
    logical_not(_flag, _flag)
    while len(_F):
        _u = _aux.take(_F)
        _i = _perm.take(_F)
        
        # This is the only bit that comes out of code generation!
        v[_u] += w[_synapses[_i]] # not 100% certain this is correct, but something like this
        
        _F += 1
        _F = extract(_flag.take(_F), _F)


GPU code: TODO!

General considerations
======================

Optimisation: read and write
----------------------------

Note that for C++ code, we can use the structure of the statements for
optimisations. For example, suppose that we only read the values of V and
do not write to them, then instead of doing this::

	double &V = _arr_V[i];
	
we can do this::

	const double V = _arr_V[i];
	
In many cases this will be much more efficient (particularly on GPU).
So we want to analyse what gets read and written to at each statement. The
current codegen2 framework does some but not all of this.

We have a similar issue in Python if we are looking at subsets of an array,
for example if we have a complicated reset function, where say variable x is
used several times, but never written to, for example::

	V = x*x*x # artificial
	
We don't want to do this::

	V[spikes] = x[spikes]*x[spikes]*x[spikes]
	
but would rather do something like this::

	x_spikes = x[spikes]
	V[spikes] = x_spikes*x_spikes*x_spikes
	
However, if x does change in a series of statements, then we can't do this
uncritically, we need to be aware of when it changes, and after it is written
to we need to update x_spikes (assuming it's used again).

Optimisation: common sub expressions
------------------------------------

Some optimising compilers will generate these automatically, but certainly not
for Python, and in my experience typically optimising compilers are not very
good at this. However, we have a potential advantage in that users will often
use equations (i.e. x = f(...)) in their Equations to simplify complex
expressions, and so we can use this as a hint for common sub-expressions.

Data types (32 vs 64 bit)
-------------------------

We should maybe be aware of datatypes, like int, float, double, etc. I propose
that int versus scalar versus bool is a good general category, with both int 
and scalar being further
subdivided. Note that there is a particular issue for 64 versus 32 bit
architectures, in that the default integer type and pointer types will be
different for these.

Also note that for C++/GPU code generation we need to know the dtypes at the
time of generating the code (i.e. we can't wait for runtime information).

Extensibility
-------------

We need to be able to allow user extensions, preferably using the same methods
as we use internally. For example, the TimedArray class can be used in
expressions as if it were just a function, but it is not a function. In order
to operate correctly it has to make sure that various names are in the
namespace, and will probably have to do some initialisation code in the case
of C++ and GPU. So we should include some hooks to allow extensibility in this
way.

GPU
---

We should bear in mind that GPU algorithms could look radically different from
their CPU counterparts, particularly in the case of Synapses. So we don't want
a code generation framework that is overly constraining on the output it
generates.

Also note that for GPU we will only have a single namespace, and we won't pass
values to the kernels by function argument (because there is a limit to how
many arguments you can pass and it's quite small). This requires setting up
some auxiliary functions to handle setting some pointers and so forth. Some
of this is already dealt with in the current codegen2.gpu framework.
