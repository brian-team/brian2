Models and neuron groups
========================
.. sidebar:: For Brian 1 users

    See the document :doc:`../introduction/brian1_to_2/neurongroup` for details how
    to convert Brian 1 code.

.. contents::
    :local:
    :depth: 1

Model equations
---------------

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

Noise
-----

In addition to ordinary differential equations, Brian allows you to
introduce random noise by specifying a
`stochastic differential equation <https://en.wikipedia.org/wiki/Stochastic_differential_equation>`__.
Brian uses the physicists' notation used in the
`Langevin equation <https://en.wikipedia.org/wiki/Langevin_equation>`__,
representing the "noise" as a term :math:`\xi(t)`, rather than the
mathematicians' stochastic differential :math:`\mathrm{d}W_t`. The
following is an example of the
`Ornstein-Uhlenbeck process <http://www.scholarpedia.org/article/Stochastic_dynamical_systems#Ornstein-Uhlenbeck_process>`__
that is often used to model a leaky integrate-and-fire neuron with
a stochastic current::

    G = NeuronGroup(10, 'dv/dt = -v/tau + sigma*xi*tau**-0.5 : volt')

You can start by thinking of ``xi`` as just a Gaussian random variable
with mean 0 and standard deviation 1. However, it scales in an
unusual way with time and this gives it units of ``1/sqrt(second)``.
You don't necessarily need to understand why this is, but it is
possible to get a reasonably simple intuition for it by thinking
about numerical integration: :ref:`see below <time_scaling_of_noise>`.

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

You can also create non-spike events. See :doc:`/advanced/custom_events`
for more details.
                     
Refractoriness
--------------
To make a neuron non-excitable for a certain time period after a spike, the
refractory keyword can be used::

    G = NeuronGroup(10, 'dv/dt = -v/tau : volt', threshold='v > -50*mV',
                    reset='v = -70*mV', refractory=5*ms)    

This will not allow any threshold crossing for a neuron for 5ms after a spike.
The refractory keyword allows for more flexible refractoriness specifications,
see :doc:`refractoriness` for details.

.. _state_variables:

State variables
---------------
Differential equations and parameters in model descriptions are stored as 
*state variables* of the `NeuronGroup`. They can be accessed and set as an
attribute of the group. To get the values without physical units (e.g. for
analysing data with external tools), use an underscore after the name:

.. doctest::

    >>> G = NeuronGroup(10, '''dv/dt = -v/tau : volt
    ...                        tau : second''')
    >>> G.v = -70*mV
    >>> G.v
    <neurongroup.v: array([-70., -70., -70., -70., -70., -70., -70., -70., -70., -70.]) * mvolt>
    >>> G.v_  # values without units
    <neurongroup.v_: array([-0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07])>

The value of state variables can also be set using string expressions that can
refer to units and external variables, other state variables, mathematical
functions, and a special variable ``i``, the index of the neuron:

.. doctest::

    >>> G.tau = '5*ms + (1.0*i/N)*5*ms'
    >>> G.tau
    <neurongroup.tau: array([ 5. ,  5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5]) * msecond>

You can also set the value only if a condition holds, for example:

.. doctest::

    >>> G.v['tau>7.25*ms'] = -60*mV
    >>> G.v
    <neurongroup.v: array([-70., -70., -70., -70., -70., -60., -60., -60., -60., -60.]) * mvolt>

Subgroups
---------
It is often useful to refer to a subset of neurons, this can be achieved using
Python's slicing syntax::

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

Here ``G1`` refers to the first 5 neurons in G, and ``G2`` to the second 5
neurons. In general ``G[i:j]`` refers to the neurons with indices from ``i``
to ``j-1``, as in general in Python.
Subgroups can be used in most places where regular groups are used, e.g. their
state variables or spiking activity can be recorded using monitors, they can be
connected via `Synapses`, etc. In such situations, indices (e.g. the indices of
the neurons to record from in a `StateMonitor`) are relative to the subgroup,
not to the main group

.. admonition:: The following topics are not essential for beginners.

    |

.. _shared_variables:

Shared variables
----------------

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

By default, subexpressions are re-evaluated whenever they are used, i.e. using
a subexpression is completely equivalent to substituting it. Sometimes it is
useful to instead only evaluate a subexpression once and then use this value
for the rest of the time step. This can be achieved by using the
``(constant over dt)`` flag. This flag is mandatory for subexpressions that
refer to stateful functions like ``rand()`` which notably allows them to be
recorded with a `StateMonitor` -- otherwise the monitor would record a different
instance of the random number than the one that was used in the equations.

For shared variables, setting by string expressions can only refer to shared values:

.. doctest::

    >>> G.shared_input = '(4.0/N)*mV'
    >>> G.shared_input
    <neurongroup.shared_input: 0.4 * mvolt>

.. _storing_state_variables:

Storing state variables
-----------------------

Sometimes it can be convenient to access multiple state variables at once, e.g.
to set initial values from a dictionary of values or to store all the values of
a group on disk. This can be done with the
:meth:`~brian2.groups.group.VariableOwner.get_states` and
:meth:`~brian2.groups.group.VariableOwner.set_states` methods:

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

The data (without physical units) can also be exported/imported to/from
`Pandas <http://pandas.pydata.org/>`_ data frames (needs an installation of ``pandas``)::

    >>> df = group.get_states(units=False, format='pandas')
    >>> df
       N      dt  i    t   tau    v
    0  5  0.0001  0  0.0  0.01  0.0
    1  5  0.0001  1  0.0  0.02  1.0
    2  5  0.0001  2  0.0  0.01  2.0
    3  5  0.0001  3  0.0  0.02  3.0
    4  5  0.0001  4  0.0  0.01  4.0
    >>> df['tau']
    0    0.01
    1    0.02
    2    0.01
    3    0.02
    4    0.01
    Name: tau, dtype: float64
    >>> df['tau'] *= 2
    >>> group.set_states(df[['tau']], units=False, format='pandas')
    >>> group.tau
    <neurongroup.tau: array([ 20.,  40.,  20.,  40.,  20.]) * msecond>


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


.. _time_scaling_of_noise:

Time scaling of noise
---------------------

Suppose we just
had the differential equation

:math:`dx/dt=\xi`

To solve this
numerically, we could compute

:math:`x(t+\mathrm{d}t)=x(t)+\xi_1`

where :math:`\xi_1` is a normally distributed random number
with mean 0 and standard deviation 1.
However, what happens if we change the time step? Suppose we used
a value of :math:`\mathrm{d}t/2` instead of :math:`\mathrm{d}t`.
Now, we compute

:math:`x(t+\mathrm{d}t)=x(t+\mathrm{d}t/2)+\xi_1=x(t)+\xi_2+\xi_1`

The mean value of :math:`x(t+\mathrm{d}t)` is 0 in both cases,
but the standard deviations are different. The first method
:math:`x(t+\mathrm{d}t)=x(t)+\xi_1` gives :math:`x(t+\mathrm{d}t)`
a standard deviation of 1, whereas the second method
:math:`x(t+\mathrm{d}t)=x(t+\mathrm{d}/2)+\xi_1=x(t)+\xi_2+\xi_1`
gives :math:`x(t)` a variance of 1+1=2 and therefore a
standard deviation of :math:`\sqrt{2}`.

In order to solve this
problem, we use the rule
:math:`x(t+\mathrm{d}t)=x(t)+\sqrt{\mathrm{d}t}\xi_1`, which makes
the mean and standard deviation of the value at time :math:`t`
independent of :math:`\mathrm{d}t`.
For this to make sense dimensionally, :math:`\xi` must have
units of ``1/sqrt(second)``.

For further details, refer to a textbook on stochastic
differential equations.
