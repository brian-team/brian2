Changes from Brian 1
====================

Major interface changes
-----------------------

Removed classes
~~~~~~~~~~~~~~~

Connections, STDP --> Synapses
MultiStateMonitor --> StateMonitor
FloatClock --> 
EventClock --> (no longer necessary)
Reset and Refractoriness classes (VariableReset, CustomRefractoriness, etc.) --> all replaced by the new string-based reset and refractory mechanisms
**Models**
leaky_IF, perfect_IF, exp_IF, quadratic_IF, Brette_Gerstner, Izhikevich, AdaptiveReset, aEIF
currents
alpha_synapse, etc.
OrnsteinUhlenbeck

Units
~~~~~

The unit system now extends to arrays, e.g. ``np.arange(5) * mV`` will retain
the units of volts and not discard them as Brian 1 did. Brian 2 is therefore
also more strict in checking the units. For example, if the state variable
``v`` uses the unit of volt, the statement ``G.v = np.rand(len(G)) / 1000.``
would now raise an error. For consistency, units are returned everywhere, e.g.
in monitors. If ``mon`` records a state variable v, ``mon.t`` will return a
time in seconds and ``mon.v`` the stored values of ``v`` in units of volts.

If a pure numpy array without units is needed for further processing, there
are several options: if it is a state variable or a recorded variable in a
monitor, appending an underscore will refer to the variable values without
units, e.g. ``mon.t_`` returns pure floating point values. Alternatively, the
units can be removed by diving through the unit (e.g. ``mon.t / second``) or
by explicitly converting it (``np.asarray(mon.t)``).


Monitors
~~~~~~~~

Changes in the internal processing
----------------------------------


