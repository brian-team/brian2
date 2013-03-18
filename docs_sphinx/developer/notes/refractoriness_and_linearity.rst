Refractoriness and linearity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Refractoriness
==============

Inactive state variables shouldn't
be integrated. This means that we need to take two paths though the state
update code depending on whether the variable is active or inactive. This can
be done in several ways.

Current Brian method
--------------------

The current Brian method is to integrate normally and
then reset certain state variables back to a fixed, given value. This is
reasonably simple and efficient, but not very flexible. This mechanism could be
updated to work with the new scheme by recording the value of each inactive
state variable when it enters the refractory state, and copying all of these
values after the integration step while they are still refractory. The code
for this would be something like::

	def on_reset(spikes):
		... (user reset code)
		clamped_values[:, spikes] = S[inactive_variables, spikes]
		
	def on_update(refractory_indices):
		... (numerical integration code)
		S[inactive_variables, refractory_indices] = clamped_values[:, refractory_indices]
		
This method has the following properties:

* Storage requirements are higher, because you need to maintain a copy of the
  clamped values of the inactive variables for all neurons. However, state
  variable storage typically isn't a bottleneck.
  
* The method is mathematically slightly wrong, because making a variable
  actually inactive would change the state updater, not just reset the values
  after the integration step. For example, an RK or other multistep method would
  use intermediate step values of an inactive variable, whereas actually they
  should be treated as constant.

* The method is nicely susceptible to vectorisation in Python, C++ (using
  SSE instructions), and GPU because you always do the same thing (no
  branching). There is some wasted computation because you compute the state
  update for all neurons even though some are refractory, and then do an
  additional unnecessary copy for these neurons. The method is instruction and
  memory cache efficient for the first stage, and the second stage is just a
  memory copy so is not too painful.

Modify equations
----------------

An alternative is to replace the following definition of a variable that is
inactive during refractoriness::

	dx/dt = f(x)
	
with this::

	dx/dt = is_refractory*f(x)
	
The only thing that is necessary then is to add an is_refractory variable and
update it when necessary. This method has the following properties:

* Storage requirements are slightly higher because the is_refractory variable
  needs to be stored, but this probably needs to be stored in every case.
  Otherwise, storage requirements are the same.
  
* This method makes it slightly more difficult to select an appropriate
  state updater. For example, linear state updaters wouldn't be possible with
  this method. However, there are ways around this, see the section on
  linear differential equations below.
  
* The method is memory and instruction cache efficient and highly vectorisable.
  However, there will be wasted computations in many cases because you will
  compute 0*complicated_expression whenever a neuron is inactive. These
  wasted instructions may not be so bad though, because refractoriness is
  not used in the case of Hodgkin-Huxley style neurons which are the most
  complicated to integrate. In the case of LIF type neurons with refractoriness,
  the computational cost is probably more associated to memory bandwidth than
  arithmetical, and this method is nice for both memory and instruction caches.

Multi-state updater method
--------------------------

We produce two state updaters, one for the active state and one
for the refractory state. Now take the state S, compute S[active], integrate
that with the active state updater, and integrate S[refractory] with the
refractory state updater, before copying the values in S[active] and
S[refractory] back to the original S. This works and is simple to implement for
code generation (because you don't even need to know it is happening):

* Storage requirements are high because you essentially need to maintain two
  copies of the state matrix. However, as above, storage is usually not the
  bottleneck.
  
* This method is mathematically elegant because it allows you to select the best
  integration method for each of the cases active/refractory.
  
* The integration phase is memory and instruction cache efficient, but there
  are two unnecessary memory copies.
  
A variant on this method is to evaluate the state update code explicitly on a
subset of the neurons. This has the following properties:

* Storage requirements are low for C++, but still high for Python.

* The method is still mathematically correct.

* For Python, the method is equivalent to the method above, so the memory and
  instruction cache use is efficient but there is an unnecessary memory copy.
  For C++ or GPU, it is instruction cache efficient but not efficient for the
  memory cache because you would work on a subset of the neurons, i.e. not
  reading and writing to memory in a contiguous fashion. This is particularly
  bad for the GPU. 

Allowing if statements in abstract code
---------------------------------------

The idea here would to be have abstract code of the form::

	if active:
		... (active state update code)
	else:
		... (refractory state update code)
		
For C++ and Python, this is straightforward to implement. For Python it is
trickier, however in that case it can probably be made essentially equivalent
to the previous idea. For example, you could do something like this::

	# if active:
	V = _array_V[active]
	... active state update code with V
	_array_V[active] = V
	# else:
	V = _array_V[-active]
	... refractory state update code with V
	_array_V[-active] = V
	
This method has the following properties:

* Storage requirements are low for C++ and high for Python.

* The method is mathematically correct.

* For Python, it is equivalent to the previous method in terms of efficiency,
  but for C++ it is slightly different. Now the method is inefficient for
  instruction cache, but efficient for memory cache. In addition, it now
  becomes difficult or impossible to use vectorised instruction sets on CPU.
  For GPU, it would introduce an if statement, but the cost of this may be
  less bad than the cost of doing two kernel launches. In terms of efficiency,
  it would be a good idea to benchmark this approach against the other
  approaches.
  
* This method has an additional benefit, in that users could now use the
  if statement in their code, which allows for potentially many interesting
  things which would have been difficult before. I think it even allows for
  nested if statements! I think this is potentially a big advantage, and is
  worth considering even if we decide not to use if statements for state update
  code.

In addition to if statements, we could actually also provide while statements.
They wouldn't be much more complicated to implement in code generation, and
would allow the specification of integration schemes which use iteration until
a certain condition is achieved, for example. This wouldn't even be too costly
for Python if we implemented it correctly, for example it could look something
like this::

	while x<10:
		x *= 2
		
would be transformed to the following in Python::

	_orig_x = x
	_cond = x<10
	_ind = arange(N)
	while sum(_cond):
		x = x[_cond]
		_ind = _ind[_cond]
		x *= 2
		_orig_x[_ind] = x
		_cond = x<10
	x = _orig_x


Linear differential equations
=============================

We want to allow for linearity in equations to be utilised. There are several
potential issues arising around this. Ideally, we would like to be able to
treat optimally each of the following cases:

* The differential equation is straightforwardly linear, i.e. dx/dt=Mx+b.
  In this case we can compute a fixed update matrix and constant so x->Ux+c.
  
* The differential equation is linear for each neuron, but the update matrix
  is different for each neuron, e.g. dV/dt=-V/tau where tau is constant but
  different for each neuron, declared with ``tau : second (const)``. In this
  case we need to compute an update matrix for each neuron.
  
* The differential equation is linear but the constants can change, e.g.
  dv/dt=-v/tau where tau is constant but can (infrequently) change. This is
  currently not handled by any of the syntax we have proposed so far, but
  we could add a new declaration, e.g. ``tau : second (rarely changing)``.
  In this case, we would need to update the matrix whenever tau changed.
  There are several potential uses of this: (1) we could use it with the
  'modify equations' state updater method above to allow linear DEs to be
  included. We would simply update the matrix for those neurons which changed
  from active to refractory or vice versa. This wouldn't be too inefficient
  because we would only update the matrix coefficients for the neurons which
  had changed when they change states, which is relatively rarely compared to
  the number of update steps. We could even cache the values for the active
  and inactive states so that it would be a simple memory copy infrequently.
  (2) it would allow people to run a simulation, change a constant, and then
  rerun it without having to recreate the NeuronGroup, Synapses, etc. This
  isn't so bad for the NeuronGroup, but it's expensive to recreate Synapses.
  
* There are mixed linear and nonlinear parts of the differential equations,
  in which case it would be nice to exactly solve the linear parts. Perhaps this
  isn't very important though? Are we even sure that it gives correct results?
  
In terms of solving these equations, the current methods in Brian are:

* S = dot(U, S)+const for standard differential equations where the matrix is
  the same for each neuron.

* MultiLinearStateUpdater for the case where the matrix is different for each
  neuron. It may now be possible to cover this case with a 3D matrix and
  multiply using einsum rather than dot, although I haven't looked into this.
  
Another possibility is to write abstract code, e.g.::

	x1 = c11*x1+c12*x2
	x2 = c21*x1+c22*x2
	
The advantage of dot compared to this is that it is highly optimised in numpy
and can make use of particular details of your processor architecture, multiple
cores, etc. However, I think these optimisations only make a big difference for
larger matrices than we are considering. The dimensions need to be big enough
that the matrix doesn't fit in a cache line, but in our case I think it pretty
much always will do. See below for benchmarking.

Finally, we might like to consider introducing a new type of parameter in
Equations, a scalar/global parameter, a single value for the whole group
rather than varying per neuron. Then variables like t and dt would be treated
in the same way as these.

Benchmarking for solving linear equations via weave and numpy.dot is
investigated in /dev/brian2/ideas/linear_state_updater_with_codegen.py. Some
results from my Windows machine (64 bit, but running 32 bit Python) are
included below. The algorithm 'dot' is the standard one in Brian. The copydot
algorithm is the standard Brian one, but a copy of the state matrix is made
from the rows of the state matrix, which could be useful if you were not
storing state variables in a matrix but in a dict of 1D arrays. The algorithm
weave is using scipy.weave, passing the update matrix, weaveopt is the same
but 0 values in the matrix are skipped, weaveopt2 directly inserts the values
into the generated source code. The algorithm 'python' uses numpy operations
but not dot (it would be what was generated by the code generation framework
that generated the weave version), and 'numexpr' is the Python version but
using numexpr. In addition, there are the weavemulti and einsum algorithms which
handle the case where there is a different matrix for each neuron. These are
slightly slower than their equivalents, but not much so.

::

	N: 10
	numsteps: 100000
	With dot: 0.61
	With copydot: 0.88 (1.4x slower)
	With weave: 0.22 (2.7x faster)
	With weavemulti: 0.40 (1.5x faster)
	With weaveopt: 0.22 (2.8x faster)
	With weaveopt2: 0.22 (2.8x faster)
	With python: 5.27 (8.6x slower)
	With numexpr: 10.19 (16.7x slower)
	
	N: 100
	numsteps: 100000
	With dot: 0.72
	With copydot: 0.99 (1.4x slower)
	With weave: 0.28 (2.5x faster)
	With weavemulti: 0.47 (1.5x faster)
	With weaveopt: 0.26 (2.8x faster)
	With weaveopt2: 0.23 (3.1x faster)
	With python: 5.51 (7.6x slower)
	With numexpr: 10.71 (14.9x slower)
	
	N: 1000
	numsteps: 100000
	With dot: 1.95
	With copydot: 2.17 (1.1x slower)
	With weave: 0.95 (2.1x faster)
	With weavemulti: 1.12 (1.7x faster)
	With weaveopt: 0.60 (3.3x faster)
	With weaveopt2: 0.39 (5.1x faster)
	With python: 7.78 (4.0x slower)
	With numexpr: 13.70 (7.0x slower)
	
	N: 10000
	numsteps: 10000
	With dot: 1.15
	With copydot: 1.06 (1.1x faster)
	With einsum: 1.65 (1.4x slower)
	With weave: 0.74 (1.6x faster)
	With weavemulti: 0.75 (1.5x faster)
	With weaveopt: 0.39 (3.0x faster)
	With weaveopt2: 0.18 (6.4x faster)
	With python: 2.97 (2.6x slower)
	With numexpr: 3.94 (3.4x slower)
	
	N: 100000
	numsteps: 1000
	With dot: 2.54
	With copydot: 2.97 (1.2x slower)
	With weave: 0.72 (3.5x faster)
	With weavemulti: 1.67 (1.5x faster)
	With weaveopt: 0.37 (6.9x faster)
	With weaveopt2: 0.15 (17.3x faster)
	With python: 5.71 (2.2x slower)
	With numexpr: 1.71 (1.5x faster)
	
	N: 1000000
	numsteps: 100
	With dot: 2.87
	With copydot: 3.66 (1.3x slower)
	With weave: 0.77 (3.7x faster)
	With weavemulti: 1.68 (1.7x faster)
	With weaveopt: 0.41 (7.0x faster)
	With weaveopt2: 0.30 (9.5x faster)
	With python: 7.01 (2.4x slower)
	With numexpr: 1.60 (1.8x faster)

The copydot is only slightly slower than the dot algorithm, meaning we can
be free to implement state variable memory in a dict of arrays rather than
forcing it to be stored in rows of a matrix. The weave versions are always
faster than the dot versions, meaning that we can take the codegen approach
easily, at least in the case of C++. However, the Python codegen versions are
all much slower, and so we would have to special case the code generation
framework in the case of pure Python + linear DE.

Providing state update code directly
====================================

An additional proposal that would be straightforward with the new code
generation mechanism would be to allow users to specify abstract code directly
to update the state variables, this would allow the specification of models
that are not based on differential equations but rather difference equations
or anything the user would like.

With this mechanism, the NeuronGroup object would simply convert the Equations
into abstract code and append it to what the user had provided. From then on,
the internal Brian code would only need to know about abstract code and not
about Equations at all.

Linked variables
================

One option we might consider is to store state variables as a dict of
(name, 1D array) pairs instead of a matrix with each state corresponding to a
row. The downside of this is that it makes linear state updaters a bit more
complicated to implement, although it seems it doesn't cost much in terms of
efficiency. The benefit is that we can now implement linked variables by
using the same memory for both variables rather than copying it each time step.
This is better for efficiency and because you are guaranteed that all changes
in one variable are instantly reflected in the linked variable. Overall, this
could be particularly beneficial for Synapses, which at least at the moment
uses linked_var internally (although it may not in Brian 2.0).
