Monitors (Brian 1 --> 2 conversion)
===================================
.. sidebar:: Brian 2 documentation

    For the main documentation about recording network activity, see the
    document :doc:`../../user/recording`.

.. contents::
    :local:
    :depth: 1

Monitoring spiking activity
---------------------------
The main class to record spiking activity is `SpikeMonitor` which is created in
the same way as in Brian 1. However, the internal storage and retrieval of
spikes is different. In Brian 1, spikes were stored as a list of pairs
``(i, t)``, the index and time of each spike. In Brian 2, spikes are stored as
two arrays ``i`` and ``t``, storing the indices and times. You can access these
arrays as attributes of the monitor, there's also a convenience attribute ``it``
that returns both at the same time. The following table shows how the spike
indices and times can be retrieved in various forms in Brian 1 and Brian 2:

+-----------------------------------------------+-------------------------------------------+
| Brian 1                                       | Brian 2                                   |
+===============================================+===========================================+
+ .. code::                                     | .. code::                                 |
+                                               |                                           |
+   mon = SpikeMonitor(group)                   |   mon = SpikeMonitor(group)               |
+   #... do the run                             |   #... do the run                         |
+   list_of_pairs = mon.spikes                  |   list_of_pairs = zip(*mon.it)            |
+   index_list, time_list = zip(*list_of_pairs) |   index_list = list(mon.i)                |
+   index_array = array(index_list)             |   time_list = list(mon.t)                 |
+   time_array = array(time_list)               |   index_array, time_array = mon.i, mon.t  |
+   # time_array is unitless in Brian 1         |   # time_array has units in Brian 2       |
+-----------------------------------------------+-------------------------------------------+

You can also access the spike times for individual neurons. In Brian 1, you
could directly index the monitor which is no longer allowed in Brian 2.
Instead, ask for a dictionary of spike times and index the returned dictionary:

+-----------------------------------------------+-----------------------------------------------+
| Brian 1                                       | Brian 2                                       |
+===============================================+===============================================+
+ .. code::                                     | .. code::                                     |
+                                               |                                               |
+   # dictionary of spike times for each neuron:|   # dictionary of spike times for each neuron:|
+   spike_dict = mon.spiketimes                 |   spike_dict = mon.spike_trains()             |
+   # all spikes for neuron 3:                  |   # all spikes for neuron 3:                  |
+   spikes_3 = spike_dict[3] #  (no units)      |   spikes_3 = spike_dict[3]  # with units      |
+   spikes_3 = mon[3] #  alternative (no units) |                                               |
+                                               |                                               |
+-----------------------------------------------+-----------------------------------------------+

In Brian 2, `SpikeMonitor` also provides the functionality of the Brian 1
classes ``SpikeCounter`` and ``PopulationSpikeCounter``. If you are only
interested in the counts and not in the individual spike events, use
``record=False`` to save the memory of storing them:

+-----------------------------------------------+-----------------------------------------------+
| Brian 1                                       | Brian 2                                       |
+===============================================+===============================================+
+ .. code::                                     | .. code::                                     |
+                                               |                                               |
+   counter = SpikeCounter(group)               |   counter = SpikeMonitor(group, record=False) |
+   pop_counter = PopulationSpikeCounter(group) |                                               |
+   #... do the run                             |   #... do the run                             |
+   # Number of spikes for neuron 3:            |   # Number of spikes for neuron 3             |
+   count_3 = counter[3]                        |   count_3 = counter.count[3]                  |
+   # Total number of spikes:                   |   # Total number of spikes:                   |
+   total_spikes = pop_counter.nspikes          |   total_spikes = counter.num_spikes           |
+                                               |                                               |
+-----------------------------------------------+-----------------------------------------------+


Currently Brian 2 provides no functionality to calculate statistics such as
correlations or histograms online, there is no equivalent to the following
classes that existed in Brian 1: ``AutoCorrelogram``, ``CoincidenceCounter``,
``CoincidenceMatrixCounter``, ``ISIHistogramMonitor``, ``VanRossumMetric``.
You will therefore have to be calculate the corresponding statistiacs manually
after the simulation based on the information stored in the `SpikeMonitor`. If
you use the default :ref:`runtime`, you can also create a new Python class that
calculates the statistic online
(see this `example from a Brian 2 tutorial <https://github.com/brian-team/brian-material/blob/master/2015-CNS-tutorial/04-advanced-brian2/coincidence_counter.ipynb>`_).


Monitoring variables
--------------------
Single variables are recorded with a `StateMonitor` in the same way as in
Brian 1, but the times and variable values are accessed differently:

+---------------------------------------+--------------------------------------+
| Brian 1                               | Brian 2                              |
+=======================================+======================================+
+ .. code::                             | .. code::                            |
+                                       |                                      |
+   mon = StateMonitor(group, 'v',      |   mon = StateMonitor(group, 'v',     |
+                      record=True)     |                      record=True)    |
+   # ... do the run                    |   # ... do the run                   |
+   # plot the trace of neuron 3:       |   # plot the trace of neuron 3:      |
+   plot(mon.times/ms, mon[3]/mV)       |   plot(mon.t/ms, mon[3].v/mV)        |
+   # plot the traces of all neurons:   |   # plot the traces of all neurons:  |
+   plot(mon.times/ms, mon.values.T/mV) |   plot(mon.t/ms, mon.v.T/mV)         |
+                                       |                                      |
+---------------------------------------+--------------------------------------+

Further differences:

* `StateMonitor` now records in the ``'start'`` scheduling slot by default. This
  leads to a more intuitive correspondence between the recorded times and the
  values: in Brian 1 (where `StateMonitor` recorded in the ``'end'`` slot) the
  recorded value at 0ms was not the initial value of the variable but the value
  after integrating it for a single time step. The disadvantage of this new
  default is that the very last value at the end of the last time step of a
  simulation is not recorded anymore. However, this value can be manually added
  to the monitor by calling `StateMonitor.record_single_timestep`.
* To not record every time step, use the ``dt`` argument (as for all other
  classes) instead of specifying a number of ``timesteps``.
* Using ``record=False`` does no longer provide mean and variance of the
  recorded variable.

In contrast to Brian 1, `StateMonitor` can now record multiple variables and
therefore replaces Brian 1's ``MultiStateMonitor``:

+-----------------------------------------------------------+------------------------------------------------------+
| Brian 1                                                   | Brian 2                                              |
+===========================================================+======================================================+
+ .. code::                                                 | .. code::                                            |
+                                                           |                                                      |
+   mon = MultiStateMonitor(group, ['v', 'w'],              |   mon = StateMonitor(group, ['v', 'w'],              |
+                           record=True)                    |                      record=True)                    |
+   # ... do the run                                        |   # ... do the run                                   |
+   # plot the traces of v and w for neuron 3:              |   # plot the traces of v and w for neuron 3:         |
+   plot(mon['v'].times/ms, mon['v'][3]/mV)                 |   plot(mon.t/ms, mon[3].v/mV)                        |
+   plot(mon['w'].times/ms, mon['w'][3]/mV)                 |   plot(mon.t/ms, mon[3].w/mV)                        |
+                                                           |                                                      |
+-----------------------------------------------------------+------------------------------------------------------+

To record variable values at the times of spikes, Brian 2 no longer provides a
separate class as Brian 1 did (``StateSpikeMonitor``). Instead, you can use
`SpikeMonitor` to record additional variables (in addition to the neuron index
and the spike time):

+-----------------------------------------------------------+------------------------------------------------------+
| Brian 1                                                   | Brian 2                                              |
+===========================================================+======================================================+
+ .. code::                                                 | .. code::                                            |
+                                                           |                                                      |
+   # We assume that "group" has a varying threshold        |   # We assume that "group" has a varying threshold   |
+   mon = StateSpikeMonitor(group, 'v')                     |   mon = SpikeMonitor(group, variables='v')           |
+   # ... do the run                                        |   # ... do the run                                   |
+   # plot the mean v at spike time for each neuron         |   # plot the mean v at spike time for each neuron    |
+   mean_values = [mean(mon.values('v', idx))               |   values = mon.values('v')                           |
+                   for idx in range(len(group))]           |   mean_values = [mean(values[idx])                   |
+                                                           |                  for idx in range(len(group))]       |
+   plot(mean_values/mV, 'o')                               |   plot(mean_values/mV, 'o')                          |
+                                                           |                                                      |
+-----------------------------------------------------------+------------------------------------------------------+

Note that there is no equivalent to ``StateHistogramMonitor``, you will have to
calculate the histogram from the recorded values or write your own custom
monitor class.
