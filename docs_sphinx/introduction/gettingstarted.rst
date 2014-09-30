Getting started with Brian
==========================

If you are familiar with Brian 1 and you're looking to get started with Brian 2,
see the section :ref:`for_brian1_users` below. For everyone else, read on.

New users
---------

In this section, we will give a very brief overview of how Brian works. We
recommend that after reading this you read through the :doc:`../user/index`
and try out some of the :doc:`../examples/index`.

All Brian scripts start with importing the Brian package::

    from brian2 import *
    
Once this has been imported, you will have access to all the Brian objects
and functions. This includes the unit system, designed to specify values with
physical dimensions like volts, amps, etc. For example, try this in an
IPython console::

    >>> from brian2 import *
    >>> print 1000*mV
    1.0 V
    >>> print 0.001*volt
    1.0 mV
    
This system is designed to stop you from inadvertently making errors either
with the scale of units (e.g. entering a value in mV when it should have been
in volts) as well as writing dimensionally inconsistent statements, e.g. the
following raises an error::

    >>> from brian2 import *
    >>> print 1*volt+1*amp
    ...
    DimensionMismatchError: Addition, dimensions were (m^2 kg s^-3 A^-1) (A)

The two core concepts in Brian are as follows:

* Groups of neurons are defined by a `NeuronGroup`, which consists of
  differential equations defining the evolution of the model, as well as
  equations specifying the "threshold condition" for a spike and the
  "reset statement" defining what happens after a spike.
* Synapses are defined by a `Synapses` object. This consists of (1) equations
  defining the evolution of the variables - same as for neurons,
  (2) equations defining what happens when a presynaptic or postsynaptic
  neuron fires a spike, (3) the pattern/structure of the synaptic connectivity
  (i.e. which neurons are connected via a synapse to which other neurons).
  
In addition, there are objects for putting input stimuli into a simulation, as
well as objects for recording the activity of a network (e.g. the spikes
produced or the time evolution of a particular variable).

A simple example that demonstrates this::

    from brian2 import *
    N = 1000
    tau = 10*ms
    vr = -70*mV
    vt = -60*mV
    eqs = '''
    dv/dt = -v/tau : volt
    '''
    G = NeuronGroup(N, eqs, threshold='v>vt', reset='v=vr')
    
In this example we have defined a group of ``N = 1000`` neurons which behave as
leaky integrate and fire neurons with instantaneous firing if the membrane
potential increases over ``vr = -60 mV`` followed by a reset to 
``vt = -70 mV``. The equations string ``dv/dt = -v/tau : volt`` gives the
group ``G`` a single variable ``v`` that evolves according to the
differential equation, and specifies that the physical unit of the variable
is the volt.

A more detailed example that actually shows some interesting behaviour can be
seen in the :doc:`CUBA example <../examples/cuba>`. For more information,
see the :doc:`../user/index`.

.. _for_brian1_users:

Brian 1 users
-------------

In most cases, Brian 2 works in a very similar way to Brian 1 but there are
some important differences to be aware of. The major distinction is that
in Brian 2 you need to be more explicit about the definition of your
simulation in order to avoid inadvertent errors. For example, the equations
defining thresholds, resets and refractoriness have to be fully explicitly
specified strings. In addition, some cases where you could use the
'magic network' system in Brian 1 won't work in Brian 2 and you'll get an
error telling you that you need to create an explicit `Network` object.

The old system of ``Connection`` and related synaptic objects such as
``STDP`` and ``STP`` have been removed and replaced with the new
`Synapses` class.

A slightly technical change that might have a significant impact on your code
is that the way 'namespaces' are handled has changed. You can now change the
value of parameters specified outside of equations between simulation runs,
as well as changing the ``dt`` value of the simulation between runs.

The units system has also been modified so that now arrays have a unit instead
of just single values. Finally, a number of objects and classes have been
removed or simplified.

For a full list of changes see :doc:`changes`.