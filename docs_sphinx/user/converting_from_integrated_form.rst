.. _integrated_form:

Converting from integrated form to ODEs
=======================================

Brian requires models to be expressed as systems of first order ordinary differential equations,
and the effect of spikes to be expressed as (possibly delayed) one-off changes. However, many
neuron models are given in *integrated form*. For example, one form of the Spike Response Model
(SRM; Gerstner and Kistler 2002) is defined as

.. math::

    V(t) = \sum_i w_i \sum_{t_i} \mathrm{PSP}(t-t_i)+V_\mathrm{rest}

where :math:`V(t)` is the membrane potential, :math:`V_\mathrm{rest}` is the rest potential,
:math:`w_i` is the synaptic weight of synapse :math:`i`, and :math:`t_i` are the timings of
the spikes coming from synapse :math:`i`, and PSP is a postsynaptic potential function.

An example PSP is the :math:`\alpha`-function :math:`\mathrm{PSP}(t)=(t/\tau)e^{-t/\tau}`.
For this function, we could rewrite the equation above in the following ODE form:

.. math::

    \tau \frac{\mathrm{d}V}{\mathrm{d}t} & = V_\mathrm{rest}-V+g \\
    \tau \frac{\mathrm{d}g}{\mathrm{d}t} &= -g \\
    g &\leftarrow g+w_i\;\;\;\mbox{upon spike from synapse $i$}

This could then be written in Brian as::

    eqs = '''
    dV/dt = (V_rest-V+g)/tau : 1
    dg/dt = -g/tau : 1
    '''
    G = NeuronGroup(N, eqs, ...)
    ...
    S = Synapses(G, G, 'w : 1', on_pre='g += w')

To see that these two formulations are the same, you first solve the problem for the case of
a single synapse and a single spike at time 0. The initial conditions at :math:`t=0` will be
:math:`V(0)=V_\mathrm{rest}`, :math:`g(0)=w`.

To solve these equations, let's substitute :math:`s=t/\tau` and take derivatives with respect to
:math:`s` instead of :math:`t`, set :math:`u=V-V_\mathrm{rest}`, and assume :math:`w=1`.
This gives us the equations :math:`u^\prime=g-u`, :math:`g^\prime=-g` with initial conditions
:math:`u(0)=0`, :math:`g(0)=1`. At this point, you can either consult a textbook on solving
linear systems of differential equations, or just
`plug this into Wolfram Alpha <https://www.wolframalpha.com/input/?i=u%27(s)%3Dg(s)-u(s),+g%27(s)%3D-g(s),+u(0)%3D0,+g(0)%3D1>`_
to get the solution :math:`g(s)=e^{-s}`, :math:`u(s)=se^{-s}` which is equal to the PSP
given above.

Now we use the linearity of these differential equations to see that it also works when
:math:`w\neq 0` and for summing over multiple spikes at different times.

In general, to convert from integrated form to ODE form, see
`Köhn and Wörgötter (1998) <http://www.mitpressjournals.org/doi/abs/10.1162/089976698300017061>`_,
`Sánchez-Montañás (2001) <https://link.springer.com/chapter/10.1007/3-540-45720-8_14>`_,
and `Jahnke et al. (1999) <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.20.2284&rep=rep1&type=pdf>`_.
However, for some simple and widely used types of synapses, use the list below. In this list, we assume synapses
are postsynaptic potentials, but you can replace :math:`V(t)` with a current or conductance for postsynaptic
currents or conductances. In each case, we give the Brian code with unitless variables, where ``eqs`` is the
differential equations for the target `NeuronGroup`, and ``on_pre`` is the argument to `Synapses`.

**Exponential synapse** :math:`V(t)=e^{-t/\tau}`::

    eqs = '''
    dV/dt = -V/tau : 1
    '''
    on_pre = 'V += w'

**Alpha synapse** :math:`V(t)=(t/\tau)e^{-t/\tau}`::

    eqs = '''
    dV/dt = (x-V)/tau : 1
    dx/dt = -x/tau    : 1
    '''
    on_pre = 'x += w'

:math:`V(t)` reaches a maximum value of :math:`w/e` at time :math:`t=\tau`.

**Biexponential synapse** :math:`V(t)=p^{-1} \left(e^{-t/\tau_1}-e^{-t/\tau_2}\right)`,
where :math:`p = \frac{\tau_1}{\tau_2}^{\frac{\tau_2}{\tau_2 - \tau_2}} - \frac{\tau_1}{\tau_2}^{\frac{\tau_1}{\tau_2 - \tau_2}}` is a normalization factor::

    eqs = '''
    dV/dt = ((tau_2 / tau_1) ** (tau_1 / (tau_2 - tau_1))*x-V)/tau_1 : 1
    dx/dt = -x/tau_2                                                 : 1
    '''
    on_pre = 'x += w'

:math:`V(t)` reaches a maximum value of :math:`w` at time
:math:`t=\frac{\tau_1\tau_2}{\tau_2-\tau_1}\log\left(\frac{\tau_2}{\tau_1}\right)`.

**STDP**

The weight update equation of the standard STDP is also often stated in an integrated form and can be
converted to an ODE form. This is covered in
:doc:`Tutorial 2 </resources/tutorials/2-intro-to-brian-synapses>`.
