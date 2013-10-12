Changes from Brian 1
====================

Major interface changes
-----------------------

More explicit model specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A design principle of Brian 2 is that model specifications are unambiguous and
explicit. Some "guessing" has therefore been removed, for example Brian 2 no
longer tries to determine which variable is the membrane potential and should
be used for thresholding and reset. This entails:

* Threshold and reset have to use explicit string descriptions, e.g.
  ``threshold='v>-50*mv'`` and ``reset='v = -70*mV'`` instead of
  ``threshold=-50*mV`` and ``reset=-70*mV``
* When a variable should be clamped during refractoriness (in Brian 1, the
  membrane potential was clamped by default), it has to be explicitly marked
  with the flag ``(unless refractory)`` in the equations
* An object such as `NeuronGroup` either uses an explicitly specified `Clock`
  or the `defaultclock` instead of using a clock defined in the current
  execution frame, if it exists

Removed classes
~~~~~~~~~~~~~~~

Several classes have been merged or are replaced by string-based model
specifications:

* *Connections* and  *STDP* are replaced by `Synapses`
* All reset and refractoriness classes (*VariableReset*,
  *CustomRefractoriness*, etc.) are replaced by the new string-based reset
  and refractoriness mechanisms, see :doc:`../user/models` and
  :doc:`../user/refractoriness`
* `Clock` is the only class for representing clocks, *FloatClock* and
  *EventClock* are obsolete
* The functionality of *MultiStateMonitor* is provided by the standard
  `StateMonitor` class.
* The library of models has been removed (*leaky_IF*, *Izhikevich*,
  *alpha_synapse*, *OrnsteinUhlenbeck*, etc.), specify the models directly
  in the equations instead

Units
~~~~~

The unit system now extends to arrays, e.g. ``np.arange(5) * mV`` will retain
the units of volts and not discard them as Brian 1 did. Brian 2 is therefore
also more strict in checking the units. For example, if the state variable
``v`` uses the unit of volt, the statement ``G.v = np.rand(len(G)) / 1000.``
will now raise an error. For consistency, units are returned everywhere, e.g.
in monitors. If ``mon`` records a state variable v, ``mon.t`` will return a
time in seconds and ``mon.v`` the stored values of ``v`` in units of volts.

If a pure numpy array without units is needed for further processing, there
are several options: if it is a state variable or a recorded variable in a
monitor, appending an underscore will refer to the variable values without
units, e.g. ``mon.t_`` returns pure floating point values. Alternatively, the
units can be removed by diving through the unit (e.g. ``mon.t / second``) or
by explicitly converting it (``np.asarray(mon.t)``).

State monitor
~~~~~~~~~~~~~

The `StateMonitor` has a slightly changed interface and also includes the
functionality of the former *MultiStateMonitor*. The stored values are accessed
as attributes, e.g.::

    mon = StateMonitor(G, ['v', 'w'], record=True)
    print mon[0].v  # v value for the first neuron, with units
    print mon.w_  # v values for all neurons, without units
    print mon. t / ms  # stored times

If accessed without index (e.g. ``mon.v``), the stored values are returned as a
two-dimensional array with the size NxM, where N is the number of recorded
neurons and M the number of time points. Therefore, plotting all values can
be achieved by::

    plt.plot(mon.t / ms, mon.v.T)

The monitor can also be indexed to give the values for a specific neuron, e.g.
``mon[0].v``. Note that in case that not all neurons are recorded, writing
``mon[i].v`` and ``mon.v[i]`` makes a difference: the former returns the value
for neuron i while the latter returns the value for the *ith* recorded neuron.::

    mon = StateMonitor(G, 'v', record=[0, 2, 4])
    print mon[2].v  # v values for neuron number 2
    print mon.v[2]  # v values for neuron number 4

Miscellaneous changes
~~~~~~~~~~~~~~~~~~~~~
* New preferences system (see :doc:`../developer/preferences`)
* New handling of namespaces (see :doc:`../user/equations`)
* New "magic" and clock system (see :doc:`../developer/new_magic_and_clocks`)
* New refractoriness system (see :doc:`../user/refractoriness`)
* More powerful string expressions that can also be used as indices for state
  variables (see e.g. :doc:`../user/synapses`)

Changes in the internal processing
----------------------------------

In Brian 1, the internal state of some objects changed when a network was run
for the first time and therefore some fundamental settings (e.g. the clock's dt,
or some code generation settings) were only taken into account before that
point. In Brian 2, objects do not change their internal state, instead they
recreate all necessary data structures from scratch at every run. This allows
to change external variables, a clock's dt, etc. between runs. Note that
currently this is not optimized for performance, i.e. some work is
unnecessarily done several times, the setup phase of a network and of each
individual run may therefore appear slow compared to Brian 1 (see #124).
