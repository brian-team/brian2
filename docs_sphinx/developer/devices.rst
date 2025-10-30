Devices
=======

This document describes how to implement a new `Device` for Brian. This is a
somewhat complicated process, and you should first be familiar with devices
from the user point of view (:doc:`/user/computation`) as well as the code
generation system (:doc:`codegen`).

We wrote Brian's devices system to allow for two major use cases, although
it can potentially be extended beyond this. The two use cases are:

1. Runtime mode. In this mode, everything is managed by Python, including
   memory management (using numpy by default) and running the simulation.
   Actual computational work can be carried out in several different ways,
   including numpy or Cython.
2. Standalone mode. In this mode, running a Brian script leads to generating
   an entire source code project tree which can be compiled and run
   independently of Brian or Python.

Runtime mode is handled by `RuntimeDevice` and is already implemented, so here
I will mainly discuss standalone devices. A good way to understand these
devices is to look at the implementation of `CPPStandaloneDevice` (the only
one implemented in the core of Brian). In many cases, the simplest way to
implement a new standalone device would be to derive a class from
`CPPStandaloneDevice` and overwrite just a few methods.

Memory management
-----------------

Memory is managed primarily via the `Device.add_array`, `Device.get_value` and
`Device.set_value` methods. When a new array is created, the `~Device.add_array`
method is called, and when trying to access this memory the other two are
called. The `RuntimeDevice` uses numpy to manage the memory and returns the
underlying arrays in these methods. The `CPPStandaloneDevice` just stores
a dictionary of array names but doesn't allocate any memory. This information
is later used to generate code that will allocate the memory, etc.

Code objects
------------

As in the case of runtime code generation, computational work is done by
a collection of `CodeObject` s. In `CPPStandaloneDevice`, each code object
is converted into a pair of ``.cpp`` and ``.h`` files, and this is probably
a fairly typical way to do it.

Building
--------

The method `Device.build` is used to generate the project. This can be
implemented any way you like, although looking at `CPPStandaloneDevice.build`
is probably a good way to get an idea of how to do it.

Device override methods
-----------------------

Several functions and methods in Brian are decorated with the `device_override`
decorator. This mechanism allows a standalone device to override the behaviour
of any of these functions by implementing a method with the name provided to
`device_override`. For example, the `CPPStandaloneDevice` uses this to
override `Network.run` as `CPPStandaloneDevice.network_run`.

Other methods
-------------

There are some other methods to implement, including initialising arrays,
creating spike queues for synaptic propagation. Take a look at the source code
for these.
