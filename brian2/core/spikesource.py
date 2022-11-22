__all__ = ["SpikeSource"]


class SpikeSource(object):
    """
    A source of spikes.

    An object that can be used as a source of spikes for objects such as
    `SpikeMonitor`, `Synapses`, etc.

    The defining properties of `SpikeSource` are that it should have:

    * A length that can be extracted with ``len(obj)``, where the maximum spike
      index possible is ``len(obj)-1``.
    * An attribute `spikes`, an array of ints each from 0 to
      ``len(obj)-1`` with no repeats (but possibly not in sorted order). This
      should be updated each time step.
    * A `clock` attribute, this will be used as the default clock for objects
      with this as a source.

    .. attribute:: spikes

        An array of ints, each from 0 to ``len(obj)-1`` with no repeats (but
        possibly not in sorted order). Updated each time step.

    .. attribute:: clock

        The clock on which the spikes will be updated.
    """

    # No implementation, just used for documentation purposes.
    pass
