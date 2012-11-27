Groups
======
A Group is a homogeneous collection of things with state variables:

* NeuronGroup is a set of neurons
* Synapses is a set of synapses
* SpatialNeuron is a set of compartments

These have a large part of functionality in common. Additionally,
complexities with device memory could be handled at this level.
This is quite substantially wider than the current Group class.

Data structure
--------------
* State variables, defined by strings, and stored as an array or
  dynamic array (could be on device)
* Units (internal Brian machinery)
* An update clock
* Potentially, a state updater (Brian machinery)

Methods
-------
* Everything related to dealing with equations (units checking etc).
* Potentially, creating a state updater (abstract state update scheme).
* Assigning variables (including with).
* Dealing with static equations.
* Perhaps dealing with linked variables.
* Perhaps creating subgroups.

Specificities
--------------
One big difference between Synapses and NeuronGroup is that Synapses has
a dynamic number of elements, implemented as dynamic array in Python.
This means for example that one can create synapses, but not neurons.

Then NeuronGroup, a subclass of group, will deal with threshold and reset.
Synapses will deal with spike-triggered events from source and target groups.
