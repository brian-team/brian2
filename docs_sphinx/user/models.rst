Models and neuron groups
========================

The core of every simulation is a `NeuronGroup`, a group of neurons that share
the same equations defining their properties. The minimum `NeuronGroup`
specification contains the number of neurons and the model description in the
form of equations::

    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) : volt')

This defines a group of 10 leaky integrators. The model description can be
directly given as a (possibly multi-line) string as above, or as an
`Equations` object. For more details on the form of equations, see
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

Sometimes it can also be useful to introduce shared variables or subexpressions,
i.e. variables that have a common value for all neurons. In contrast to
external variables (such as ``Cm`` above), such variables can change during a
run, e.g. by using :meth:`~brian2.groups.group.Group.run_regularly`. This can be
for example used for an external stimulus that changes in the course of a run::

    G = NeuronGroup(10, '''shared_input : volt (shared)
                           dv/dt = (-v + shared_input)/tau : volt
                           tau : second''')

Note that there are several restrictions around the use of shared variables:
they cannot be written to in contexts where statements apply only to a subset
of neurons (e.g. reset statements, see below). If a code block mixes statements
writing to shared and vector variables, then the shared statements have to
come first.

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
attribute of the group. To get the values without physical units (e.g. for
analysing data with external tools), use an underscore after the name:

.. doctest::

    >>> G = NeuronGroup(10, '''dv/dt = (-v + shared_input)/tau : volt
                               shared_input : volt (shared)
    ...                        tau : second''')
    >>> G.v = -70*mV
    >>> print G.v
    <neurongroup.v: array([-70., -70., -70., -70., -70., -70., -70., -70., -70., -70.]) * mvolt>
    >>> print G.v_  # values without units
    <neurongroup.v_: array([-0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07])>
    >>> G.shared_input = 5*mV
    >>> print G.shared_input
    <neurongroup.shared_input: 5.0 * mvolt>

The value of state variables can also be set using string expressions that can
refer to units and external variables, other state variables, mathematical
functions, and a special variable ``i``, the index of the neuron:

.. doctest::

    >>> G.tau = '5*ms + 5*ms*rand() + i*5*ms'
    >>> print G.tau
    <neurongroup.tau: array([  5.03593449,  10.74914808,  19.01641896,  21.66813281,
            27.16243388,  31.13571924,  36.28173038,  40.04921519,
            47.28797921,  50.18913711]) * msecond>

For shared variables, such string expressions can only refer to shared values:

.. doctest::

    >>> G.shared_input = 'rand()*mV + 4*mV'
    >>> print G.shared_input
    <neurongroup.shared_input: 4.2579690100000001 * mvolt>

Sometimes it can be convenient to access multiple state variables at once, e.g.
to set initial values from a dictionary of values or to store all the values of
a group on disk. This can be done with the `Group.get_states` and
`Group.set_states` methods:

.. doctest::

    >>> group = NeuronGroup(5, '''dv/dt = -v/tau : 1
    ...                           tau : second''')
    >>> initial_values = {'v': [0, 1, 2, 3, 4],
    ...                   'tau': [10, 20, 10, 20, 10]*ms}
    >>> group.set_states(initial_values)
    >>> group.v[:]
    array([ 0.,  1.,  2.,  3.,  4.])
    >>> group.tau[:]
    array([ 10.,  20.,  10.,  20.,  10.]) * msecond
    >>> states = group.get_states()
    >>> states['v']
    array([ 0.,  1.,  2.,  3.,  4.])
    >>> sorted(states.keys())
    ['N', 'dt', 'i', 't', 'tau', 'v']


Subgroups
---------
It is often useful to refer to a subset of neurons, this can be achieved using
slicing syntax::

    G = NeuronGroup(10, '''dv/dt = -v/tau : volt
                           tau : second''',
                    threshold='v > -50*mV',
                    reset='v = -70*mV')
    # Create subgroups
    G1 = G[:5]
    G2 = G[5:]

    # This will set the values in the main group, subgroups are just "views"
    G1.tau = 10*ms
    G2.tau = 20*ms

Subgroups can be used in most places where regular groups are used, e.g. their
state variables or spiking activity can be recorded using monitors, they can be
connected via `Synapses`, etc. In such situations, indices (e.g. the indices of
the neurons to record from in a `StateMonitor`) are relative to the subgroup,
not to the main group


.. _linked_variables:

Linked variables
----------------
A `NeuronGroup` can define parameters that are not stored in this group, but are
instead a reference to a state variable in another group. For this, a group
defines a parameter as ``linked`` and then uses `linked_var` to
specify the linking. This can for example be useful to model shared noise
between cells::

    inp = NeuronGroup(1, 'dnoise/dt = -noise/tau + tau**-0.5*xi : 1')

    neurons = NeuronGroup(100, '''noise : 1 (linked)
                                  dv/dt = (-v + noise_strength*noise)/tau : volt''')
    neurons.noise = linked_var(inp, 'noise')

If the two groups have the same size, the linking will be done in a 1-to-1
fashion. If the source group has the size one (as in the above example) or if
the source parameter is a shared variable, then the linking will be done as
1-to-all. In all other cases, you have to specify the indices to use for the
linking explicitly::

    # two inputs with different phases
    inp = NeuronGroup(2, '''phase : 1
                            dx/dt = 1*mV/ms*sin(2*pi*100*Hz*t-phase) : volt''')
    inp.phase = [0, pi/2]

    neurons = NeuronGroup(100, '''inp : volt (linked)
                                  dv/dt = (-v + inp) / tau : volt''')
    # Half of the cells get the first input, other half gets the second
    neurons.inp = linked_var(inp, 'x', index=repeat([0, 1], 50))


.. _numerical_integration:

Numerical integration
---------------------
Differential equations are converted into a sequence of statements that
integrate the equations numerically over a single time step. By default, Brian
chooses an integration method automatically, trying to solve the equations
exactly first (for linear equations) and then resorting to numerical algorithms.
It will also take care of integrating stochastic differential equations
appropriately. Each class defines its own list of algorithms it tries to
apply, `NeuronGroup` and `Synapses` will use the first suitable method out of
the methods ``'linear'``, ``'euler'`` and ``'heun'`` while `SpatialNeuron`
objects will use ``'linear'``, ``'exponential_euler'``, ``'rk2'`` or ``'heun'``.

If you prefer to chose an integration algorithm yourself, you can do so using
the ``method`` keyword for `NeuronGroup`, `Synapses`, or `SpatialNeuron`.
The complete list of available methods is the following:

* ``'linear'``: exact integration for linear equations
* ``'independent'``: exact integration for a system of independent equations,
  where all the equations can be analytically solved independently
* ``'exponential_euler'``: exponential Euler integration for conditionally
  linear equations
* ``'euler'``: forward Euler integration (for additive stochastic
  differential equations using the Euler-Maruyama method)
* ``'rk2'``: second order Runge-Kutta method (midpoint method)
* ``'rk4'``: classical Runge-Kutta method (RK4)
* ``'heun'``: stochastic Heun method for solving Stratonovich stochastic
  differential equations with non-diagonal multiplicative noise.
* ``'milstein'``: derivative-free Milstein method for solving stochastic
  differential equations with diagonal multiplicative noise

You can also define your own numerical integrators, see
:doc:`../advanced/state_update` for details.
