Refractoriness
==============

Internally, a `NeuronGroup` with refractoriness has a boolean variable
`not_refractory` added to the equations, and this is used to implement
the refractoriness behaviour. Specifically, the ``threshold`` condition
is replaced by ``threshold and not_refractory`` and differential equations
that are marked as ``(unless refractory)`` are multiplied by
``int(not_refractory)`` (so that they have the value 0 when the neuron is
refractory).

This ``not_refractory`` variable is also available to the user
to define more sophisticated refractoriness behaviour.
For example, the following code updates the
``w`` variable with a different time constant during refractoriness::

    G = NeuronGroup(N, '''dv/dt = -(v + w)/ tau_v : 1 (unless refractory)
                          dw/dt = (-w / tau_active)*int(not_refractory) + (-w / tau_ref)*(1 - int(not_refractory)) : 1''',
                    threshold='v > 1', reset='v=0; w+=0.1', refractory=2*ms)
