Input stimuli
=============

There are various ways of providing "external" input to a network. Brian2 does
not yet provide all the features of Brian1 in this regard, but there is already
a range of options, detailed below.

Poisson input
-------------
For generating spikes according to a Poisson point process, `PoissonGroup` can
be used. It takes a rate or an array of rates (one rate per neuron) as an
argument and can be connected to a `NeuronGroup` via the usual `Synapses`
mechanism. At the moment, using ``PoissonGroup(N, rates)`` is equivalent to
``NeuronGroup(N, 'rates : Hz', threshold='rand()*dt<rates')`` and setting the
group's ``rates`` attribute. The explicit creation of such a `NeuronGroup` might
be useful if the rates for the neurons are not constant in time, since it allows
using the techniques mentioned below (formulating rates as equations or
referring to a timed array). In the future, the implementation of `PoissonGroup`
will change to a more efficient spike generation mechanism, based on the
calculation of inter-spike intervals. Note that, as can seen in its equivalent
`NeuronGroup` formulation, a `PoissonGroup` does not work for high rates where
more than one spike might fall into a single timestep. Use several units with
lower rates in this case (e.g. use ``PoissonGroup(10, 1000*Hz)`` instead of
``PoissonGroup(1, 10000*Hz)``).

Example use::

    P = PoissonGroup(100, np.arange(100)*Hz + 10*Hz)
    G = NeuronGroup(100, 'dv/dt = -v / (10*ms) : 1')
    S = Synapses(P, G, pre='v+=0.1', connect='i==j')

Explicit equations
------------------
If the input can be explicitly expressed as a function of time (e.g. a
sinusoidal input current), then its description can be directly included in
the equations of the respective group::

    G = NeuronGroup(100, '''dv/dt = (-v + I)/(10*ms) : 1
                            rates : Hz  # each neuron's input has a different rate
                            size : 1  # and a different amplitude
                            I = size*sin(2*pi*rates*t) : 1''')
    G.rates = '10*Hz + i*Hz'
    G.size = '(100-i)/100. + 0.1'


Timed arrays
------------
If the time dependence of the input cannot be expressed in the equations in the
way shown above, it is possible to create a `TimedArray`. Such an objects acts
as a function of time where the values at given time points are given
explicitly. This can be especially useful to describe non-continuous
stimulation. For example, the following code defines a `TimedArray` where
stimulus blocks consist of a constant current of random strength for 30ms,
followed by no stimulus for 20ms. Note that in this particular example,
numerical integration can use exact methods, since it can assume that the
`TimedArray` is a constant function of time during a single integration time
step. Also note that the semantics of `TimedArray` changed slightly compared
to Brian1: for ``TimedArray([x1, x2, ...], dt=my_dt)``, the value ``x1`` will be
returned for all ``0<=t<my_dt``, ``x2`` for ``my_dt<=t<2*my_dt`` etc., whereas
Brian1 returned ``x1`` for ``0<=t<0.5*my_dt``,
``x2`` for ``0.5*my_dt<=t<1.5*my_dt``, etc.

::

    stimulus = TimedArray(np.hstack([[c, c, c, 0, 0]
                                     for c in np.random.rand(1000)]),
                                    dt=10*ms)
    G = NeuronGroup(100, 'dv/dt = (-v + 2*stimulus(t))/(10*ms) : 1',
                    threshold='v>1', reset='v=0')
    G.v = '0.5*rand()'  # different initial values for the neurons

Abstract code statements
------------------------
An alternative to specifying a stimulus in advance is to run a series of
abstract code statements at certain points during a simulation. This can be
achieved with a `CodeRunner`, one can think of these statements as equivalent
to reset statements but executed unconditionally (i.e. for all neurons) and
possibly on a different clock as the rest of the group. The following code
changes the stimulus strength of half of the neurons (randomly chosen) to a new
random value every 50ms. Note that the statement uses logical expressions to
have the values only updated for the chosen subset of neurons (where the
newly introduced auxiliary variable ``change`` equals 1)::

  G = NeuronGroup(100, '''dv/dt = (-v + I)/(10*ms) : 1
                          I : 1  # one stimulus per neuron''')
  stim_updater = G.runner('''change = int(rand() < 0.5)
                             I = change*(rand()*2) + (1-change)*I''',
                          when=Scheduler(clock=Clock(dt=50*ms), when='start'))


Arbitrary Python code (network operations)
------------------------------------------
If none of the above techniques is general enough to fulfill the requirements
of a simulation, Brian allows to write a `NetworkOperation`, an arbitrary
Python function that is executed every timestep (possible on a different clock
than the rest of the simulation). This function can do arbitrary operations,
use conditional statements etc. and it will be executed as it is (i.e. as pure
Python code even if weave codegeneration is active). Note that one cannot use
network operations in combination with the C++ standalone mode. Network
operations are particularly useful when some condition or calculation depends
on operations across neurons, which is currently not possible to express in
abstract code. The following code switches input on for a randomly chosen single
neuron every 50ms::

    G = NeuronGroup(10, '''dv/dt = (-v + active*I)/(10*ms) : 1
                           I = sin(2*pi*100*Hz*t) : 1 (scalar) #single input
                           active : 1  # will be set in the network function''')
    @network_operation(when=Clock(dt=50*ms))
    def update_active():
        print defaultclock.t
        index = np.random.randint(10)  # index for the active neuron
        G.active_ = 0  # the underscore switches off unit checking
        G.active_[index] = 1

Note that the network operation (in the above example: ``update_active``) has
to be included in the `Network` object if one is constructed explicitly.
