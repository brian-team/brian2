Recording during a simulation
=============================

Recording variables during a simulation is done with "monitor" objects.
Specifically, spikes are recorded with `SpikeMonitor`, the time evolution of
variables with `StateMonitor` and the firing rate of a population of neurons
with `PopulationRateMonitor`.

Recording spikes
----------------

To record spikes from a group ``G`` simply create a `SpikeMonitor` via
``SpikeMonitor(G)``. After the simulation, you can access the attributes
``i``, ``t``, ``it``, ``num_spikes`` and ``count`` of the monitor.
The ``i`` and ``t``
attributes give the array of neuron indices and times of the spikes. For
example, if ``M.i==[0, 2, 1]`` and ``M.t==[1*ms, 2*ms, 3*ms]`` it means that
neuron 0 fired a spike at 1 ms, neuron 2 fired a spike at 2 ms, and neuron 1
fired a spike at 3 ms. The ``num_spikes`` attribute gives the total number
of spikes recorded, and ``count`` is an array of the length of the recorded
group giving the total number of spikes recorded from each neuron. Finally, 
the ``it`` attribute is just the pair ``(i, t)`` for convenience.

Example::

    G = NeuronGroup(...)
    M = SpikeMonitor(G)
    run(...)
    plot(M.t/ms, M.i, '.')

Recording variables
-------------------

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
