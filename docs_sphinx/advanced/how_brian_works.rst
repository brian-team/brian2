How Brian works
===============

In this section we will briefly cover some of the internals of how Brian
works. This is included here to understand the general process that Brian
goes through in running a simulation, but it will not be sufficient to
understand the source code of Brian itself or to extend it to do new things.
For a more detailed view of this, see the documentation in the
:doc:`../developer/index`

The user-visible part of Brian consists of a number of objects such as
`NeuronGroup`, `Synapses`, `Network`, etc. These are all written in pure
Python and essentially work to translate the user specified model into the
computational engine. The end state of this translation is a collection of
short blocks of code operating on a namespace, which are called
in a sequence by the `Network`. Examples of these short blocks of code are the
"state updaters" which perform numerical integration, or the synaptic
propagation step. The namespaces consist of a mapping from names to values,
where the possible values can be scalar values, fixed-length or dynamically
sized arrays, and functions.

Syntax layer
------------

The syntax layer consists of everything that is independent of the way the
final simulation is computed (i.e. the language and device it is running on).
This includes things like `NeuronGroup`, `Synapses`, `Network`, `Equations`,
etc.

The user-visible part of this is documented fully in the :doc:`../user/index`
and the :doc:`../advanced/index`. In particular, things such as the analysis
of equations and assignment of numerical integrators. The end result of this
process, which is passed to the computational engine, is a specification of
the simulation consisting of the following data:

* A collection of variables which are scalar values, fixed-length arrays,
  dynamically sized arrays, and functions. These are handled by `Variable`
  objects detailed in :doc:`../developer/variables_indices`. Examples:
  each state variable of a `NeuronGroup` is assigned an `ArrayVariable`;
  the list of spike indices stored by a `SpikeMonitor` is assigned a
  `DynamicArrayVariable`; etc.
* A collection of code blocks specified via an "abstract code block" and a
  template name. The "abstract code block" is a sequence of statements such
  as ``v = vr`` which are to be executed. In the case that say, ``v`` and
  ``vr`` are arrays, then the statement is to be executed for each element of
  the array. These abstract code blocks are either given directly by the user
  (in the case of neuron threshold and reset, and synaptic pre and post codes),
  or generated from differential equations combined with a numerical
  integrator. The template name is one of a small set (around 20 total) which
  give additional context. For example, the code block ``a = b`` when
  considered as part of a "state update" means execute that for each neuron
  index. In the context of a reset statement, it means execute it for each
  neuron index of a neuron that has spiked. Internally, these templates need
  to be implemented for each target language/device, but there are relatively
  few of them.
* The order of execution of these code blocks, as defined by the `Network`.

Computational engine
--------------------

The computational engine covers everything from generating to running code in
a particular language or on a particular device. It starts with the
abstract definition of the simulation resulting from the syntax layer
described above.

The computational engine is described by a `Device` object. This is used for
allocating memory, generating and running code. There are two types of device,
"runtime" and "standalone". In runtime mode, everything is managed by Python,
even if individual code blocks are in a different language. Memory is managed
using numpy arrays (which can be passed as pointers to use in other
languages). In standalone mode, the output of the process (after calling
`Device.build`) is a complete source code project that handles everything,
including memory management, and is independent of Python.

For both types of device, one of the key steps that works in the same way is
code generation, the creation of a compilable and runnable block of code from an
abstract code block and a collection of variables. This happens in two stages:
first of all, the abstract code block is converted into a code snippet,
which is a syntactically correct block of code in the target language, but
not one that can run on its own (it doesn't handle accessing the variables
from memory, etc.). This code snippet typically represents the inner loop code.
This step is handled by a `CodeGenerator` object. In some
cases it will involve a syntax translation (e.g. the Python syntax ``x**y`` in
C++ should be ``pow(x, y)``). The
next step is to insert this code snippet into a template to form a compilable
code block. This code block is then passed to a runtime `CodeObject`. In the
case of standalone mode, this doesn't do anything, but for runtime devices
it handles compiling the code and then running the compiled code block in the
given namespace.