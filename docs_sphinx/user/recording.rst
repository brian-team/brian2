Recording during a simulation
=============================

Recording variables during a simulation is done with "monitor" objects.
Specifically, spikes are recorded with `SpikeMonitor`, the time evolution of
variables with `StateMonitor` and the firing rate of a population of neurons
with `PopulationRateMonitor`.

Note that all monitors are implement as "groups", so you can get all the stored
values in a monitor with the `Group.get_states` method, which can be useful to
dump all recorded data to disk, for example.

Recording spikes
----------------

To record spikes from a group ``G`` simply create a `SpikeMonitor` via
``SpikeMonitor(G)``. After the simulation, you can access the attributes
``i``, ``t``, ``it``, ``num_spikes`` and ``count`` of the monitor.
The ``i`` and ``t``
attributes give the array of neuron indices and times of the spikes. For
example, if ``M.i==[0, 2, 1]`` and ``M.t==[1*ms, 2*ms, 3*ms]`` it means that
neuron 0 fired a spike at 1 ms, neuron 2 fired a spike at 2 ms, and neuron 1
fired a spike at 3 ms. Alternatively, you can also call the
`~brian2.monitors.spikemonitor.SpikeMonitor.spike_trains` method to get a
dictionary mapping neuron indices to arrays of spike times, i.e. in the above
example, ``spike_trains = M.spike_trains(); spike_trains[1]`` would return
``array([  3.]) * msecond``. The ``num_spikes`` attribute gives the total number
of spikes recorded, and ``count`` is an array of the length of the recorded
group giving the total number of spikes recorded from each neuron. Finally, the
``it`` attribute is just the pair ``(i, t)`` for convenience.

Example::

    G = NeuronGroup(N, model='...')
    M = SpikeMonitor(G)
    run(runtime)
    plot(M.t/ms, M.i, '.')

If you are only interested in summary statistics but not the individual spikes,
you can set the ``record`` argument to ``False``. You will then not have access
to ``i`` and ``t`` but you can still get the ``count`` and the total number of
spikes (``num_spikes``).

.. _recording_variables_spike_time:

Recording variables at spike time
---------------------------------

By default, a `SpikeMonitor` only records the time of the spike and the index
of the neuron that spiked. Sometimes it can be useful to addtionaly record
other variables, e.g. the membrane potential for models where the threshold is
not at a fixed value. This can be done by providing an extra ``variables``
argument, the recorded variable can then be accessed as an attribute of the
`SpikeMonitor`. To conveniently access the values of a recorded variable for
a single neuron, the `SpikeMonitor.values` method can be used that returns a
dictionary with the values for each neuron.::

    G = NeuronGroup(N, '''dv/dt = (1-v)/(10*ms) : 1
                          v_th : 1''',
                    threshold='v > v_th',
                    # randomly change the threshold after a spike:
                    reset='''v=0
                             v_th = clip(v_th + rand()*0.2 - 0.1, 0.1, 0.9)''')
    G.v_th = 0.5
    spike_mon = SpikeMonitor(G, variables='v')
    run(1*second)
    v_values = spike_mon.values('v')
    print('Threshold crossing values for neuron 0: {}'.format(v_values[0]))
    hist(spike_mon.v, np.arange(0, 1, .1))
    show()

.. note:: Spikes are not the only events that can trigger recordings, see
          :doc:`../advanced/custom_events`.

.. _recording_variables_continuously:

Recording variables continuously
--------------------------------

To record how a variable evolves over time, use a `StateMonitor`. To use this,
you specify the group, variables and indices you want to record from. You
specify the variables with a string or list of strings, and the indices
either as an array of indices or ``True`` to record all indices (but beware
because this may take a lot of memory). 

After the simulation, you can access these variables as attributes of the
monitor. They are 2D arrays with shape ``(num_indices, num_times)``. The
special attribute ``t`` is an array of length ``num_times`` with the
corresponding times at which the values were recorded.

Note that you can also use `StateMonitor` to record from `Synapses` where
the indices are the synapse indices rather than neuron indices.

In this example, we record two variables v and u, and record from indices 0,
10 and 100. Afterwards, we plot the recorded values of v and u from neuron 0::

    G = NeuronGroup(...)
    M = StateMonitor(G, ('v', 'u'), record=[0, 10, 100])
    run(...)
    plot(M.t/ms, M.v[0]/mV, label='v')
    plot(M.t/ms, M.u[0]/mV, label='u')

There are two subtly different ways to get the values for specific neurons: you
can either index the 2D array stored in the attribute with the variable name
(as in the example above) or you can index the monitor itself. The former will
use an index relative to the recorded neurons (e.g. `M.v[1]` will return the
values for the second *recorded* neuron which is the neuron with the index 10
whereas `M.v[10]` would raise an error because only three neurons have been
recorded), whereas the latter will use an absolute index corresponding to the
recorded group (e.g. `M[1].v` will raise an error because the neuron with the
index 1 has not been recorded and `M[10].v` will return the values for the
neuron with the index 10). If all neurons have been recorded (e.g. with
``record=True``) then both forms give the same result.

Note that for plotting all recorded values at once, you have to transpose the
variable values::

    plot(M.t/ms, M.v.T/mV)


In contrast to previous versions of Brian, the values are recorded at the
beginning of a time step and not at the end (you can set the ``when`` argument
when creating a `StateMonitor`, details about scheduling can be
found here: :doc:`../advanced/scheduling`).

Recording population rates
--------------------------

To record the time-varying firing rate of a population of neurons use
`PopulationRateMonitor`. After the simulation the monitor will have two
attributes ``t`` and ``rate``, the latter giving the firing rate at each
time step corresponding to the time in ``t``. For example::

    G = NeuronGroup(...)
    M = PopulationRateMonitor(G)
    run(...)
    plot(M.t/ms, M.rate/Hz)

To get a smoother version of the rate, use `PopulationRateMonitor.smooth_rate`.