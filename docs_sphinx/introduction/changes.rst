Changes from Brian 1
====================

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

For more details, see below.

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

Clocks and networks
~~~~~~~~~~~~~~~~~~~

Brian's system of handling clocks and networks has been substantially
changed. You now usually specify a value of ``dt`` either globally or
explicitly for each object rather than creating clocks (although this is
still possible).

More importantly, the behaviour of networks is different:

* Either you create a `Network` of objects you want to simulate explicitly,
  or you use the 'magic' system which now simulates all named objects in
  the context where you run it.
* The magic network will now raise errors if you try to do something where
  it cannot accurately guess what you mean. In these situations, we recommend
  using an explicit `Network`.
* Objects can now only belong to a single `Network` object, in order to avoid
  inadvertent errors.
* Similarly, you can no longer change the time explicitly: the only way the
  time changes is by running a simulation. Instead, you can `store` and
  `restore` the state of a `Network` (including the time).

Removed classes
~~~~~~~~~~~~~~~

Several classes have been merged or are replaced by string-based model
specifications:

* *Connections*, *STP* and  *STDP* are replaced by `Synapses`
* All reset and refractoriness classes (*VariableReset*,
  *CustomRefractoriness*, etc.) are replaced by the new string-based reset
  and refractoriness mechanisms, see :doc:`../user/models` and
  :doc:`../user/refractoriness`
* `Clock` is the only class for representing clocks, *FloatClock* and
  *EventClock* are obsolete
* The functionality of *MultiStateMonitor* is provided by the standard
  `StateMonitor` class.
* The functionality of *StateSpikeMonitor* is provided by the
  `SpikeMonitor` class.
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

Another change is that the `StateMonitor` now records in the ``'start'``
scheduling slot by default. This leads to a more intuitive correspondence
between the recorded times and the values: previously (where `StateMonitor`
recorded in the ``'end'`` slot) the recorded value at 0ms was not the initial
value of the variable but the value after integrating it for a single time
step. The disadvantage of this new default is that the very last value at the
end of the last time step of a simulation is not recorded anymore. However, this
value can be manually added to the monitor by calling
`StateMonitor.record_single_timestep`.

Miscellaneous changes
~~~~~~~~~~~~~~~~~~~~~
* New preferences system (see :doc:`../developer/preferences`)
* New handling of namespaces (see :doc:`../user/equations`)
* New "magic" and clock system (see :doc:`../advanced/scheduling` and
  :doc:`../user/running`)
* New refractoriness system (see :doc:`../user/refractoriness`)
* More powerful string expressions that can also be used as indices for state
  variables (see e.g. :doc:`../user/synapses`)
* "Brian Hears" is being rewritten, but there is a bridge to the version
  included in Brian 1 until the new version is written (see
  :doc:`../user/brian1hears_bridge`)
* `Equations` objects no longer save their namespace, they now behave just
  like strings.
* There is no longer any ``reinit()`` mechanism, this is now handled by
  `store` and `restore`.

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
