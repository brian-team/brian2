Simulation on multiple devices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an early stage planning document for running Brian on multiple devices.

Introduction
============

Code can be generated for different targets, e.g. C or GPU.
This means that code will be executed in different languages, but also that
memory can be stored on a external device. This would be the case of simulation
on GPU, clusters or neuromorphic devices.

In this case, arrays for NeuronGroup and Synapses should not be create on the
host but on the device. The host would only store the address on the device.
The simplest way to handle this, I think, is to have a special class
array_on_device.

We should also consider that a number of variables would probably also need
to be stored on the device (for example, time of last spike)
When we create any variable, we need to keep in mind that the device may need
to access it.

What needs to be done is to distinguish clearly between variables and objects
that are specific to Brian cooking, or needed for it (e.g. model specification,
network structure, etc), and simulation variables, which can potentially all
be on the device.
Among simulation variables, some may only need to be in the code (constant
variables, e.g. number of neurons in a group), others require a creation in
device memory.

So I suggest that, from the start, we clearly categorize all variables as
being device variables, constant variables, structural variables
(basically whether the device needs to know about it).

The sections below give a relatively complicated but highly flexible design,
and the final section gives a much simpler alternative design that would be
less flexible but might be sufficient.

Design considerations
=====================

In order to run Brian code the following steps need to be carried out:

* The models need to be defined, this includes the string definitions of
  equations, etc. This should almost certainly not be carried out on the
  device, but only in the core of Brian.

* Memory needs to be allocated. This should be carried out on the target
  device, although we should also have a default memory allocation structure
  for the reference Brian implementation that can be used by other devices if
  they have access to the CPU.

* Data values need to be initialised, i.e. initial values, connectivity, etc.
  This could be carried out in core Brian, and values transmitted to the
  device, or it could be carried out on the device itself.

* Code needs to be generated and compiled. The generation should probably be
  carried out in core Brian, and the compilation either in core Brian or on the
  device depending on details. A possible exception would be if we wanted to,
  for example, use a different bit of software to integrate the differential
  equations, e.g. PyDSTool or CVODE, in which case we wouldn't specify an
  integration method, but rather allow this software to construct the entire
  state update part of the NeuronGroup object.

* The main loop needs to be run. This can either happen entirely on the device
  or in a cooperative manner between the device and core Brian. For example,
  you might only execute part of the code on the device, and part of the code
  in core Brian (e.g. reading realtime data and preliminary analysis on a
  device, and then simulating a network in Brian, or for GPU you might only
  yield to the GPU for doing state update and propagation, but overall
  simulation control could still happen in core Brian).

* Results should be analysed, and possibly saved to a file. For this, and for
  initial values, we need a way to transmit data between core Brian and the
  device, possibly just saving to a file. But this stage is technically
  optional.

This seems to suggest we have the following aspects of a device that can be
implemented:

* Memory allocation, can use core Brian memory for CPU memory allocations.

* Direct memory communication between devices (maybe sufficient to allow core
  Brian to device communication?).

* Compilation and execution of code. The generation of the code should be done
  by the appropriate :class:`Language` object, and the device should specify
  which languages it accepts.

* Simulation control. This is either done entirely by the device or partly
  by the device and partly by core Brian.
  
In order to make the system flexible and not too difficult to implement a
device, we want to design a core set of Brian->device interactions that allow
for all of these possibilities. In order to force ourselves to keep multiple
devices compatible with all of Brian, we should implement the core of Brian
itself as a device.

In addition to making it work, we need the code not to be so complicated that
it makes Brian itself impossible to understand or develop for.

Interactions
============

The following table is a first attempt at a list of all the Brian->device
interactions that are needed to implement flexible devices. An important
overall categorisation might be the distinction between **construction time**
and **runtime interactions**.

* *Memory*

  - **Allocate memory on device**: requires a datatype and a size.
  - **Refer to memory**: something like a DevicePointer class, which can store
    anything in principle, for example it might store an actual pointer, or
    just a unique ID.
  - **Allocate dynamic memory**: needed by Synapses, but may not be implemented
    on the device, which would mean doing some of the work at construction time
    in core Brian.

* *Code*

  - **Accepted languages**: the device should specify a list
  - **Compile code**: the device should provide a method for compiling code
    and for **Referring to compiled code objects**. Code is executed in a
    namespace, and so for compiled code there should be a **Mapping from
    names in the namespace to allocated memory**.
    
* *Data*

  - **Send/receive data**: specify a source and target memory location and a
    datatype. Note that strings can be data, and potentially even code objects?
    This would be to allow the construction of values by code (e.g. setting the
    synaptic weights by a code string).
    
* *Brian objects*

  - **Implemented objects**: the device should specify which Brian objects it
    has a complete implementation of. This might need to be further broken down
    into **Runtime implementation** and **Construction**. So, if possible we
    would use both, but only the runtime implementation is necessary.
  - **Dependent interactions**: Brian objects should specify a set of
    Brian->device interactions they need to be able to function, e.g. they
    may need to be able to send/receive data, refer to memory, etc.

On running a script for a specific device, Brian would determine what the
capabilities of the device were, check if it is possible to run it on that
device, and turn each Brian call into a Brian->device interaction of some sort.
These Brian->device interactions would not necessarily have to execute
immediately, but might be cached into a sequence of instructions that are then
sent together to the device.

Flexible interactions
---------------------

The user could write the following::

	G.V = 'rand()*10*mV'
	
This could be broken down in several different ways.

1. It could be evaluated in core Brian as ``x = rand(len(G))*10*mV``, and then
   the data ``x`` could be sent at runtime to the device memory that ``G.V``
   points to. In terms of Brian->device interactions this requires:
   
   * Refer to device memory
   * Send array data to device memory target
   
2. It could be evaluated in core Brian, saved to a file, and then referred to
   at construction time only on the device.
   
3. It could be evaluated on the device. This requires a Brian->device
   interaction that allows us to specify that we want to evaluate an expression
   for an array on the device.
   
A more complicated case if if we specify the synaptic connectivity structure
via a code string. In this case, we not only have to evaluate an expression but
also manage a dynamic memory data structure.

Case studies
============

Core Brian
----------

The reference implementation, everything should be implemented and memory is
allocated on the CPU. Core Brian could be, for example, CPUDevice or something
like that.

C++ output
----------

The idea of this would be to generate complete C++ code that carried out the
simulation. This could be done either by generating data in Python, saving to
disk and then loading in the C++ file, but would ideally take place entirely in
C++ so that it would actually be independent of Brian.

This device would cache the series of statements that generated it and then
produce output code as a string or file. Optionally, it could automatically
compile and run this code, and then return data to the Python script for
analysis.

Memory allocation would be handled by storing the fact that memory needs to be
allocated, and this would be added to the generated C++ source code. References
to memory would be stored as unique symbol names or IDs in a memory manager
class (could use smart pointers or some such here). Code language would be C++
and code objects would be referred to by unique function or class names.
Sending and receiving of data would either not be possible, or only possible
via files. All objects to be used would have to be implemented on the device,
there is no possibility of runtime communication.

An interesting case to consider here is that the user script may do something
like this::

	G.V = rand(len(G))*10*mV
	
or it may do::

	G.V = 'rand()*10*mV'
	
The latter is a new concept for Brian 2.0 and already present for Synapses in
Brian 1.4. This construction-time interaction would allow it to be handled in
C++, but we should also allow the former. To enable the former, when working
on this device we would have to write the value of rand(len(G))*10*mV to a 
data file and then read this data file in the C++ source code. Note that like
this, multiple runs would produce the same data, but there's no alternative
here. A warning could be raised for the user in this case. Alternatively, we
could override rand() to return an array-like object that remembers that it was
originally random, but in more complicated scenarios this probably breaks down.
Another way to handle this is that if you are running the Brian script, and
this is both generating, compiling and running the C++ code, then even if you
wanted to cache the generation and compilation of the code, you could still
regenerate the data files each time so that it would be different each run.
This also clearly shows we need a way to handle both setting variables with
data and with code.

GPU
---

Most of the considerations here are the same as for C++ code except that we
have a bit more flexibility to have CPU->GPU runtime interactions.

Android
-------

Most of the considerations here are the same as for C++ output, except that the
language and device implementation will be different.

A simpler proposal
==================

The design above would be highly flexible, but it's also extremely complicated,
and this clashes with the overall guideline that Brian 2.0 should be easier
to understand and develop for than Brian 1.x. It's difficult to see how a fully
flexible scheme could be developed that allows memory to be located on different
devices, but still interact with the CPU, or not, etc., so this alternative
design is necessarily more limited in scope. However, it may be sufficient in
practice.

Rather than allowing parts of Brian to be implemented on the device, and other
parts not to be implemented on the device, we instead say that a "Brian device"
is a complete reimplementation of all or a part of the Brian interface. To
use a device you would write ``from brian.devices.devicename import *`` instead
of ``from brian import *``. This is similar to how it works in pyNN.
A priori, this puts a heavier burden on someone
writing a new device to work with Brian, but this may not be as bad as it seems.

With the new design for Brian 2.0, the number of core objects and classes is
greatly reduced, as these classes are all much more flexible. In addition,
the core code of Brian is much more modular, meaning that even though a device
needs to reimplement the whole interface, it can reuse many parts of the
standard Brian package to do so. For example, the code generation module would
still be available. We would provide a 'template' device that essentially
reimplements Brian using Brian, and this would serve as a base to start a
full device implementation.

As above, the output of running it would be device dependent. For C++ the
output could be a C++ project with some datafiles and source code that could
be compiled and run. Similarly for Android. In these cases, you wouldn't be
able to do analysis after calling run(), and in general access to internal
variables would be limited and would raise errors. For GPU however, we could
implement an automatic communication protocol so that you could run, transfer
data back to CPU, do analysis, then potentially transfer data back to GPU, etc.
We would also write some helper code that made it easier to do this sort of
thing.

Some of the things we were hoping to do with devices would not be possible in
this scheme, i.e. ideas for replacing just one part of Brian with another
implementation (e.g. using PyDSTool for the numerical integration). However,
now that Brian is much more modular, we have an alternative approach for this.
For example, if we have a NumericalIntegration object that we pass to
NeuronGroup, developers would only need to reimplement this part (and as before
they would have access to all of Brian's internals to do so). Normally, the
user wouldn't see this at all, because by default NeuronGroup would get a copy
of the standard Brian NumericalIntegration object, but if they wanted to use
a more accurate numerical integrator like PyDSTool or CVODE then they would
just specify this when defining the NeuronGroup.

This system makes the following requirements on Brian development, all of which
are anyway more or less in line with what we already want to do:

* Everything should be made highly modular, with it being possible to swap out
  one implementation for another. This also suggests heavy use of duck typing.
* There should be as much code re-use as possible in order to reduce the burden
  on device implementers.
  
This vision of Brian is that it could become partly a platform, with the more
traditional Brian being a reference implementation, and some additional work
being tools to make it easier to implement the Brian interface (such as codegen,
the equations module, etc.).