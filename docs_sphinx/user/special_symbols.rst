List of special symbols
=======================

The following lists all of the special symbols that Brian uses in
equations and code blocks, and their meanings.

dt
    Time step width
i
    Index of a neuron (`NeuronGroup`) or the pre-synaptic neuron
    of a synapse (`Synapses`)
j
    Index of a post-synaptic neuron of a synapse
lastspike
    Last time that the neuron spiked (for refractoriness)
lastupdate
    Time of the last update of synaptic variables in event-driven
    equations.
N
    Number of neurons (`NeuronGroup`) or synapses (`Synapses`). Use
    ``N_pre`` or ``N_post`` for the number of presynaptic or
    postsynaptic neurons in the context of `Synapses`.
not_refractory
    Boolean variable that is normally true, and false if the neuron
    is currently in a refractory state
t
    Current time
xi, xi_*
    Stochastic differential in equations
