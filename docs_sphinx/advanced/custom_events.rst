Custom events
=============

In most simulations, a `NeuronGroup` defines a threshold on its membrane
potential that triggers a spike event. This event can be monitored by a
`SpikeMonitor`, it is used in synaptic interactions, and in integrate-and-fire
models it also leads to the execution of one or more reset statements.

Sometimes, it can be useful to define additional events, e.g. when an ion
concentration in the cell crosses a certain threshold. This can be done with
the ``events`` keyword in the `NeuronGroup` initializer::

    group = NeuronGroup(N, '...', threshold='...', reset='...',
                        events={'custom_event': 'x > x_th'})

In this example, we define an event with the name ``custom_event`` that is
triggered when the ``x`` variable crosses the threshold ``x_th``. Such events
can be recorded with an `EventMonitor`::

    event_mon = EventMonitor(group, 'custom_event')

Such an `EventMonitor` can be used in the same way as a `SpikeMonitor` -- in
fact, creating the `SpikeMonitor` is basically identical to recording the
``spike`` event with an `EventMonitor`. An `EventMonitor` is not limited to
record the event time/neuron index, it can also record other variables of the
model::

    event_mon = EventMonitor(group, 'custom_event', variables['var1', 'var2'])

If the event should trigger a series of statements (i.e. the equivalent of
``reset`` statements), this can be added by calling `~NeuronGroup.run_on_event`::

    group.run_on_event('custom_event', 'x=0')

When neurons are connected by `Synapses`, the ``pre`` and ``post`` pathways
are triggered by spike events by default. It is possible to change this by
providing an ``on_event`` keyword that either specifies which event to use for all
pathways, or a specific event for each pathway (where non-specified pathways use
the default ``spike`` event)::

    synapse_1 = Synapses(group, another_group, '...', on_pre='...', on_event='custom_event')
    synapse_2 = Synapses(group, another_group, '...', on_pre='...', on_post='...',
                         on_event={'pre': 'custom_event'})

Scheduling
----------
By default, custom events are checked after the spiking threshold (in the
``after_thresholds`` slots) and statements are executed after the reset (in
the ``after_resets`` slots). The slot for the execution of custom
event-triggered statements can be changed when it is added with the usual
``when`` and ``order`` keyword arguments (see :doc:`scheduling` for details).
To change the time when the condition is checked, use
`NeuronGroup.set_event_schedule`.
