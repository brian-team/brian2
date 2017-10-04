Custom events
=============

Overview
--------

In most simulations, a `NeuronGroup` defines a threshold on its membrane
potential that triggers a spike event. This event can be monitored by a
`SpikeMonitor`, it is used in synaptic interactions, and in integrate-and-fire
models it also leads to the execution of one or more reset statements.

Sometimes, it can be useful to define additional events, e.g. when an ion
concentration in the cell crosses a certain threshold. This can be done
with the custom events system in Brian, which is illustrated in this diagram.

.. image:: custom_events.*

You can see in this diagram that the source `NeuronGroup` has four types
of events, called ``spike``, ``evt_other``, ``evt_mon`` and ``evt_run``.
The event ``spike`` is the default event. It is triggered when you
you include ``threshold='...'`` in a `NeuronGroup`, and has two
potential effects. Firstly, when the event is triggered it causes the
reset code to run, specified by ``reset='...'``. Secondly, if there
are `Synapses` connected, it causes the ``on_pre`` on ``on_post``
code to run (depending if the `NeuronGroup` is presynaptic or
postsynaptic for those `Synapses`).

In the diagram though, we have three additional event types. We've
included several event types here to make it clearer, but you could
use the same event for different purposes. Let's start
with the first one, ``evt_other``. To understand this, we need to look at
the `Synapses` object in a bit more detail. A `Synapses` object has
multiple *pathways* associated to it. By default, there are just two,
called ``pre`` and ``post``. The ``pre`` pathway is activated by
presynaptic spikes, and the ``post`` pathway by postsynaptic spikes.
Specifically, the ``spike`` event on the presynaptic `NeuronGroup` triggers
the ``pre`` pathway, and the ``spike`` event on the postsynaptic
`NeuronGroup` triggers the ``post`` pathway. In the example in the diagram,
we have created a new pathway called ``other``, and the ``evt_other``
event in the presynaptic `NeuronGroup` triggers this pathway. Note that
we can arrange this however we want. We could have ``spike`` trigger the
``other`` pathway if we wanted to, or allow it to trigger both the
``pre`` and ``other`` pathways. We could also allow ``evt_other`` to
trigger the ``pre`` pathway. See below for details on the syntax for this.

The third type of event in the example is named ``evt_mon`` and this
is connected to an `EventMonitor` which works exactly the same way
as `SpikeMonitor` (which is just an `EventMonitor` attached by default
to the event ``spike``).

Finally, the fourth type of event in the example is named ``evt_run``,
and this causes some code to be run in the `NeuronGroup` triggered by
the event. To add this code, we call `NeuronGroup.run_on_event`. So,
when you set ``reset='...'``, this is equivalent to calling
`NeuronGroup.run_on_event` with the ``spike`` event.

Details
-------

Defining an event
~~~~~~~~~~~~~~~~~

This can be done with
the ``events`` keyword in the `NeuronGroup` initializer::

    group = NeuronGroup(N, '...', threshold='...', reset='...',
                        events={'custom_event': 'x > x_th'})

In this example, we define an event with the name ``custom_event`` that is
triggered when the ``x`` variable crosses the threshold ``x_th``. Note
that you can define any number of custom events. Each event is defined
by its name as the key, and its condition as the value of the
dictionary.

Recording events
~~~~~~~~~~~~~~~~

Custom events can be recorded with an `EventMonitor`::

    event_mon = EventMonitor(group, 'custom_event')

Such an `EventMonitor` can be used in the same way as a `SpikeMonitor` -- in
fact, creating the `SpikeMonitor` is basically identical to recording the
``spike`` event with an `EventMonitor`. An `EventMonitor` is not limited to
record the event time/neuron index, it can also record other variables of the
model at the time of the event::

    event_mon = EventMonitor(group, 'custom_event', variables['var1', 'var2'])

Triggering `NeuronGroup` code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the event should trigger a series of statements (i.e. the equivalent of
``reset`` statements), this can be added by calling `~NeuronGroup.run_on_event`::

    group.run_on_event('custom_event', 'x=0')

Triggering synaptic pathways
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When neurons are connected by `Synapses`, the ``pre`` and ``post`` pathways
are triggered by ``spike`` events on the presynaptic and postsynaptic `NeuronGroup`
by default. It is possible to change which pathway is triggered by which event by
providing an ``on_event`` keyword that either specifies which event to use for all
pathways, or a specific event for each pathway (where non-specified pathways use
the default ``spike`` event)::

    synapse_1 = Synapses(group, another_group, '...', on_pre='...', on_event='custom_event')

The code above causes all pathways to be triggered by an event named ``custom_event``
instead of the default ``spike``.

::

    synapse_2 = Synapses(group, another_group, '...', on_pre='...', on_post='...',
                         on_event={'pre': 'custom_event'})

In the code above, only the ``pre`` pathway is triggered by the ``custom_event``
event.

We can also create new pathways and have them be triggered by custom events.
For example::

    synapse_3 = Synapses(group, another_group, '...',
                         on_pre={'pre': '....',
                                 'custom_pathway': '...'},
                         on_event={'pre': 'spike',
                                   'custom_pathway': 'custom_event'})

In this code, the default ``pre`` pathway is still triggered by the ``spike``
event, but there is a new pathway called ``custom_pathway`` that is triggered
by the ``custom_event`` event.

Scheduling
~~~~~~~~~~
By default, custom events are checked after the spiking threshold (in the
``after_thresholds`` slots) and statements are executed after the reset (in
the ``after_resets`` slots). The slot for the execution of custom
event-triggered statements can be changed when it is added with the usual
``when`` and ``order`` keyword arguments (see :ref:`scheduling` for details).
To change the time when the condition is checked, use
`NeuronGroup.set_event_schedule`.
