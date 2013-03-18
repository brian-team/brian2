Brian 2.0 - Project management
==============================
The goal of this document is separate the work into different independent
components.

First, I think we can distinguish between Brian internal machinery
and dealing with target/device execution. The latter is the most complex,
because we need to think about .

Brian internal machinery
------------------------
For this, we do not need to care about what the target is going to be.
We can assume that this will only be Python code, with data structures stored
on the host. This is closest to our previous work, some of which we can reuse.

Things we can reuse with minor changes:

* units
* magic
* clocks
* global preferences
* logging
* plotting

Redesigned things:

* Equations. The job of this new class is essentially to create a data
  structure.
* Code strings (expression/statements). This will include basic string manipulation
  and units checking   (although possibly as a function rather than a method). Most importantly,
  the creation of a namespace that depends on context (e.g. definition within
  a NeuronGroup or a Synapses). A lot of this is currently done in inspection.
* State updaters. This turns equations code in an update scheme. At this stage,
  this is still not specific of a target. Although it may be for specific schemes
  like linear update. So perhaps we could divide between target-independent updaters
  target-specific updaters.

Targets and devices
-------------------
Construction might be done on an external device. This means data structures
may be stored on the device rather than the host.
Simulation can be performed in an external device, and/or in a different
language than Python.
Here we must always ask ourselves: what if the network is executed and hosted
on the device?

Mostly unchanged:

* Network. The main thing that may change is if we want to output and/or
  execute code for the main loop, but this is a minor addition.

Redesigned:

* NeuronGroup. This should be rewritten almost entirely. We must take into
  account device memory.
* Synapses. This is largely done, but should be connected with code generation.
  We also need to think about device memory, but this is in fact exactly the
  same issue as for NeuronGroup. I suggest there is a base class for NeuronGroup
  and Synapses.

Completely new:

* Code generation

Inputs and monitors
-------------------
We need to think about refactoring/abstracting the current classes, if
possible. We also need to think about code generation and device memory.

General guidelines
------------------
See the specific document on style.
