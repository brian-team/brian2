Models and neuron groups
========================

The core of every simulation is a `NeuronGroup`, a group of neurons that share
the same equations defining their properties. The minimum `NeuronGroup`
specification contains the number of neurons and the model description in the
form of equations::

    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) : volt')

This defines a group of 10 leaky integrators. The model description can be
directly given as a (possibly multi-line) string as above, or as an
`Equation` object. For more details on the form of equations, see
:doc:`equations`. Note that model descriptions can make reference to physical
units, but also to scalar variables declared outside of the model description
itself::

    tau = 10*ms
    G = NeuronGroup(10, 'dv/dt = -v/tau : volt')

If a variable should be taken as a *parameter* of the neurons, i.e. if it
should be possible to vary its value across neurons, it has to be declared
as part of the model description::

    G = NeuronGroup(10, '''dv/dt = -v/tau : volt
                           tau : second''')

To make complex model descriptions more readable, named subexpressions can
be used::

    G = NeuronGroup(10, '''dv/dt = I_leak / Cm : volt
                           I_leak = g_L*(E_L - v) : amp''')

Threshold and reset
-------------------
To emit spikes, neurons need a *threshold*. Threshold and reset are given
as strings in the `NeuronGroup` constructor::

    tau = 10*ms
    G = NeuronGroup(10, 'dv/dt = -v/tau : volt', threshold='v > -50*mV',
                    reset='v = -70*mV')

Whenever the threshold condition is fulfilled, the reset statements will be
executed. Again, both threshold and reset can refer to physical units,
external variables and parameters, in the same way as model descriptions::

    v_r = -70*mV  # reset potential
    G = NeuronGroup(10, '''dv/dt = -v/tau : volt
                           v_th : volt  # neuron-specific threshold''',
                    threshold='v > v_th', reset='v = v_r')
                     
Refractoriness
--------------
To make a neuron non-excitable for a certain time period after a spike, the
refractory keyword can be used::

    G = NeuronGroup(10, 'dv/dt = -v/tau : volt', threshold='v > -50*mV',
                    reset='v = -70*mV', refractory=5*ms)    

This will not allow any threshold crossing for a neuron for 5ms after a spike.
The refractory keyword allows for more flexible refractoriness specifications,
see :doc:`refractoriness` for details.

State variables
---------------
Differential equations and parameters in model descriptions are stored as 
*state variables* of the `NeuronGroup`. They can be accessed and set as an
attribute of the group:

.. doctest::

    >>> G = NeuronGroup(10, '''dv/dt = -v/tau : volt
    ...                        tau : second''')
    >>> G.v = -70*mV
    >>> print G.v
    <neurongroup.v: array([-70., -70., -70., -70., -70., -70., -70., -70., -70., -70.]) * mvolt>

The value of state variables can also be set using string expressions that can
refer to units and external variables, other state variables, mathematical
functions, and a special variable ``i``, the index of the neuron:

.. doctest::

    >>> G.tau = '5*ms + 5*ms*rand() + i*5*ms'
    >>> print G.tau
    <neurongroup.tau: array([  5.03593449,  10.74914808,  19.01641896,  21.66813281,
            27.16243388,  31.13571924,  36.28173038,  40.04921519,
            47.28797921,  50.18913711]) * msecond>
