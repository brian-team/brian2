Overview of Brian 2.0
=====================

The aims of Brian 2.0 are:

* To simplify the structure, including getting rid of a number of classes. In particular,
  the Synapses class will replace Connection, STP and STDP.
* To refactor a number of things, in particular dealing with models defined in strings.
* To make it easier for other people to contribute, by making the structure more modular.
* To develop target-specific implementations (Python, C, GPU...).
* To have more meaningful and systematic error messages.

To achieve these goals, Brian 2.0 will rely more systematically on code generation, where
the code will depend on the target.
There are three types of code:

* State update code, defined by equations. This is used both in NeuronGroup and
  Synapses.
* Threshold code, defined as an expression. This is also used in Synapses, to
  set the values of synaptic variables.
* Reset code, defined as a series of statements. This is also used in Synapses,
  for events triggered by pre and postsynaptic spikes.

We may rephrase it as follows:

* Equations
* Expressions
* Statements

Equations are specific, and maybe they should first be turned into statements.
I suggest the following: equations consist of essentially of expressions.
The main job of the Equations class is to turn expressions into statements
(state update code).

Another cornerstone of Brian 2.0 is the use of a single class, Synapses, to
represent everything synaptic, including synaptic plasticity.
In this new class, delays are handled differently. All spikes produced by
neurons are directly sent to a SpikeQueue. Delays are thus handled in a
separate object, SpikeQueue. One implication is that the LS attribute of
neuron groups is no longer useful.
