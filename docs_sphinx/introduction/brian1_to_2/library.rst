Library models (Brian 1 --> 2 conversion)
=========================================

.. contents::
    :local:
    :depth: 1

Neuron models
-------------
The neuron models in Brian 1's ``brian.library.IF`` package are nothing more
than shorthands for equations. The following table shows how the models from
Brian 1 can be converted to explicit equations (and reset statements in the case
of the adaptive exponential integrate-and-fire model) for use in Brian 2. The
examples include a "current" ``I`` (depending on the model not necessarily in
units of Ampère) and could e.g. be used to plot the f-I curve of the neuron.

Perfect integrator
~~~~~~~~~~~~~~~~~~
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| Brian 1                                                          | Brian 2                                                                                  |
+==================================================================+==========================================================================================+
+ .. code::                                                        | .. code::                                                                                |
+                                                                  |                                                                                          |
+    eqs = (perfect_IF(tau=10*ms) +                                |    tau = 10*ms                                                                           |
+           Current('I : volt'))                                   |    eqs = '''dvm/dt = I/tau : volt                                                        |
+    group = NeuronGroup(N, eqs,                                   |             I : volt'''                                                                  |
+                        threshold='v > -50*mV',                   |    group = NeuronGroup(N, eqs,                                                           |
+                        reset='v = -70*mV')                       |                        threshold='v > -50*mV',                                           |
+                                                                  |                        reset='v = -70*mV')                                               |
+                                                                  |                                                                                          |
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+

Leaky integrate-and-fire neuron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| Brian 1                                                          | Brian 2                                                                                  |
+==================================================================+==========================================================================================+
+ .. code::                                                        | .. code::                                                                                |
+                                                                  |                                                                                          |
+    eqs = (leaky_IF(tau=10*ms, El=-70*mV) +                       |    tau = 10*ms; El = -70*mV                                                              |
+           Current('I : volt'))                                   |    eqs = '''dvm/dt = ((El - vm) + I)/tau : volt                                          |
+    group = ... # see above                                       |             I : volt'''                                                                  |
+                                                                  |    group = ... # see above                                                               |
+                                                                  |                                                                                          |
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+

Exponential integrate-and-fire neuron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| Brian 1                                                          | Brian 2                                                                                  |
+==================================================================+==========================================================================================+
+ .. code::                                                        | .. code::                                                                                |
+                                                                  |                                                                                          |
+    eqs = (exp_IF(C=1*nF, gL=30*nS, EL=-70*mV,                    |    C = 1*nF; gL = 30*nS; EL = -70*mV; VT = -50*mV; DeltaT = 2*mV                         |
+                  VT=-50*mV, DeltaT=2*mV) +                       |    eqs = '''dvm/dt = (gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT) + I)/C : volt             |
+           Current('I : amp'))                                    |             I : amp'''                                                                   |
+    group = ... # see above                                       |    group = ... # see above                                                               |
+                                                                  |                                                                                          |
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+

Quadratic integrate-and-fire neuron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| Brian 1                                                          | Brian 2                                                                                  |
+==================================================================+==========================================================================================+
+ .. code::                                                        | .. code::                                                                                |
+                                                                  |                                                                                          |
+    eqs = (quadratic_IF(C=1*nF, a=5*nS/mV,                        |    C = 1*nF; a=5*nS/mV; EL=-70*mV; VT = -50*mV                                           |
+           EL=-70*mV, VT=-50*mV) +                                |    eqs = '''dvm/dt = (a_q*(vm-EL)*(vm-VT) + I)/C : volt                                  |
+           Current('I : amp'))                                    |             I : amp'''                                                                   |
+    group = ... # see above                                       |    group = ... # see above                                                               |
+                                                                  |                                                                                          |
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+

Izhikevich neuron
~~~~~~~~~~~~~~~~~
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| Brian 1                                                          | Brian 2                                                                                  |
+==================================================================+==========================================================================================+
+ .. code::                                                        | .. code::                                                                                |
+                                                                  |                                                                                          |
+    eqs = (Izhikevich(a=0.02/ms, b=0.2/ms) +                      |    a = 0.02/ms; b = 0.2/ms                                                               |
+           Current('I : volt/second'))                            |    eqs = '''dvm/dt = (0.04/ms/mV)*vm**2+(5/ms)*vm+140*mV/ms-w + I : volt                 |
+    group = ... # see above                                       |             dw/dt = a_I*(b_I*vm-w) : volt/second                                         |
+                                                                  |             I : volt/second'''                                                           |
+                                                                  |    group = ... # see above                                                               |
+                                                                  |                                                                                          |
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+

Adaptive exponential integrate-and-fire neuron ("Brette-Gerstner model")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| Brian 1                                                          | Brian 2                                                                                  |
+==================================================================+==========================================================================================+
+ .. code::                                                        | .. code::                                                                                |
+                                                                  |                                                                                          |
+    # AdEx, aEIF, and Brette_Gerstner all refer to the same model |   C = 1*nF; gL = 30*nS; EL = -70*mV; VT = -50*mV; DeltaT = 2*mV; tauw = 150*ms; a = 4*nS |
+    eqs = (aEIF(C=1*nF, gL=30*nS, EL=-70*mV,                      |   eqs = '''dvm/dt = (gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT) -w + I)/C : volt           |
+                VT=-50*mV, DeltaT=2*mV, tauw=150*ms, a=4*nS) +    |            dw/dt=(a_BG*(vm-EL)-w)/tauw : amp                                             |
+           Current('I:amp'))                                      |            I : volt/second'''                                                            |
+    group = NeuronGroup(N, eqs,                                   |   group = NeuronGroup(N, eqs,                                                            |
+                        threshold='v > -20*mV',                   |                       threshold='v > -20*mV',                                            |
+                        reset=AdaptiveReset(Vr=-70*mV, b=0.08*nA))|                       reset='vm=-70*mV; w += 0.08*nA')                                   |
+                                                                  |                                                                                          |
+------------------------------------------------------------------+------------------------------------------------------------------------------------------+

Ionic currents
--------------
Brian 1's functions for ionic currents, provided in
``brian.library.ionic_currents`` correspond to the following equations (note
that the currents follow the convention to use a shifted membrane potential,
i.e. the membrane potential at rest is 0mV):

+-------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| Brian 1                                                                 | Brian 2                                                                          |
+=========================================================================+==================================================================================+
+ .. code::                                                               | .. code::                                                                        |
+                                                                         |                                                                                  |
+    from brian.library.ionic_currents import *                           |    defaultclock.dt = 0.01*ms                                                     |
+    defaultclock.dt = 0.01*ms                                            |    gl = 60*nS; El = 10.6*mV                                                      |
+    eqs_leak = leak_current(gl=60*nS, El=10.6*mV, current_name='I_leak') |    eqs_leak = Equations('I_leak = gl*(El - vm) : amp')                           |
+                                                                         |    g_K = 7.2*uS; EK = -12*mV                                                     |
+    eqs_K = K_current_HH(gmax=7.2*uS, EK=-12*mV, current_name='I_K')     |    eqs_K = Equations('''I_K = g_K*n**4*(EK-vm) : amp                             |
+                                                                         |                         dn/dt = alphan*(1-n)-betan*n : 1                         |
+    eqs_Na = Na_current_HH(gmax=24*uS, ENa=115*mV, current_name='I_Na')  |                         alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz   |
+                                                                         |                         betan = .125*exp(-.0125*vm/mV)/ms : Hz''')               |
+    eqs = (MembraneEquation(C=200*pF) +                                  |    g_Na = 24*uS; ENa = 115*mV                                                    |
+           eqs_leak + eqs_K + eqs+Na +                                   |    eqs_Na = Equations('''I_Na = g_Na*m**3*h*(ENa-vm) : amp                       |
+           Current('I_inj : amp'))                                       |                          dm/dt=alpham*(1-m)-betam*m : 1                          |
+                                                                         |                          dh/dt=alphah*(1-h)-betah*h : 1                          |
+                                                                         |                          alpham=.1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz   |
+                                                                         |                          betam=4*exp(-.0556*vm/mV)/ms : Hz                       |
+                                                                         |                          alphah=.07*exp(-.05*vm/mV)/ms : Hz                      |
+                                                                         |                          betah=1./(1+exp(3.-.1*vm/mV))/ms : Hz''')               |
+                                                                         |    C = 200*pF                                                                    |
+                                                                         |    eqs = Equations('''dvm/dt = (I_leak + I_K + I_Na + I_inj)/C : volt            |
+                                                                         |                       I_inj : amp''') + eqs_leak + eqs_K + eqs_Na                |
+                                                                         |                                                                                  |
+-------------------------------------------------------------------------+----------------------------------------------------------------------------------+

Synapses
--------
Brian 1's synaptic models, provided in ``brian.library.synpases`` can be
converted to the equivalent Brian 2 equations as follows:

Current-based synapses
~~~~~~~~~~~~~~~~~~~~~~
+----------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| Brian 1                                                                          | Brian 2                                                                          |
+==================================================================================+==================================================================================+
+ .. code::                                                                        | .. code::                                                                        |
+                                                                                  |                                                                                  |
+    syn_eqs = exp_current('s', tau=5*ms, current_name='I_syn')                    |    tau = 5*ms                                                                    |
+    eqs = (MembraneEquation(C=1*nF) + Current('Im = gl*(El-vm) : amp') +          |    syn_eqs = Equations('dI_syn/dt = -I_syn/tau : amp')                           |
+           syn_eqs)                                                               |    eqs = (Equations('dvm/dt = (gl*(El - vm) + I_syn)/C : volt') +                |
+    group = NeuronGroup(N, eqs, threshold='vm>-50*mV', reset='vm=-70*mV')         |           syn_eqs)                                                               |
+    syn = Synapses(source, group, pre='s += 1*nA')                                |    group = NeuronGroup(N, eqs, threshold='vm>-50*mV', reset='vm=-70*mV')         |
+    # ... connect synapses, etc.                                                  |    syn = Synapses(source, group, pre='I_syn += 1*nA')                            |
+                                                                                  |    # ... connect synapses, etc.                                                  |
+                                                                                  |                                                                                  |
+----------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
+ .. code::                                                                        | .. code::                                                                        |
+                                                                                  |                                                                                  |
+    syn_eqs = alpha_current('s', tau=2.5*ms, current_name='I_syn')                |   tau = 2.5*ms                                                                   |
+    eqs = ... # remaining code as above                                           |   syn_eqs = Equations('''dI_syn/dt = (s - I_syn)/tau : amp                       |
+                                                                                  |                          ds/dt = -s/tau : amp''')                                |
+                                                                                  |   group = NeuronGroup(N, eqs, threshold='vm>-50*mV', reset='vm=-70*mV')          |
+                                                                                  |   syn = Synapses(source, group, pre='s += 1*nA')                                 |
+                                                                                  |   # ... connect synapses, etc.                                                   |
+                                                                                  |                                                                                  |
+----------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
+ .. code::                                                                        | .. code::                                                                        |
+                                                                                  |                                                                                  |
+    syn_eqs = biexp_current('s', tau1=2.5*ms, tau2=10*ms, current_name='I_syn')   |    tau1 = 2.5*ms; tau2 = 10*ms; invpeak = (tau2 / tau1) ** (tau1 / (tau2 - tau1))|
+    eqs = ... # remaining code as above                                           |    syn_eqs = Equations('''dI_syn/dt = (invpeak*s - I_syn)/tau1 : amp             |
+                                                                                  |                           ds/dt = -s/tau2 : amp''')                              |
+                                                                                  |    eqs = ... # remaining code as above                                           |
+                                                                                  |                                                                                  |
+----------------------------------------------------------------------------------+----------------------------------------------------------------------------------+

Conductance-based synapses
~~~~~~~~~~~~~~~~~~~~~~~~~~
+----------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| Brian 1                                                                          | Brian 2                                                                          |
+==================================================================================+==================================================================================+
+ .. code::                                                                        | .. code::                                                                        |
+                                                                                  |                                                                                  |
+    syn_eqs = exp_conductance('s', tau=5*ms, E=0*mV, conductance_name='g_syn')    |    tau = 5*ms; E = 0*mV                                                          |
+    eqs = (MembraneEquation(C=1*nF) + Current('Im = gl*(El-vm) : amp') +          |    syn_eqs = Equations('dg_syn/dt = -g_syn/tau : siemens')                       |
+           syn_eqs)                                                               |    eqs = (Equations('dvm/dt = (gl*(El - vm) + g_syn*(E - vm))/C : volt') +       |
+    group = NeuronGroup(N, eqs, threshold='vm>-50*mV', reset='vm=-70*mV')         |           syn_eqs)                                                               |
+    syn = Synapses(source, group, pre='s += 10*nS')                               |    group = NeuronGroup(N, eqs, threshold='vm>-50*mV', reset='vm=-70*mV')         |
+    # ... connect synapses, etc.                                                  |    syn = Synapses(source, group, pre='g_syn += 10*nS')                           |
+                                                                                  |    # ... connect synapses, etc.                                                  |
+                                                                                  |                                                                                  |
+----------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
+ .. code::                                                                        | .. code::                                                                        |
+                                                                                  |                                                                                  |
+    syn_eqs = alpha_conductance('s', tau=2.5*ms, E=0*mV, conductance_name='g_syn')|   tau = 2.5*ms; E = 0*mV                                                         |
+    eqs = ... # remaining code as above                                           |   syn_eqs = Equations('''dg_syn/dt = (s - g_syn)/tau : siemens                   |
+                                                                                  |                          ds/dt = -s/tau : siemens''')                            |
+                                                                                  |   group = NeuronGroup(N, eqs, threshold='vm>-50*mV', reset='vm=-70*mV')          |
+                                                                                  |   syn = Synapses(source, group, pre='s += 10*nS')                                |
+                                                                                  |   # ... connect synapses, etc.                                                   |
+                                                                                  |                                                                                  |
+----------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
+ .. code::                                                                        | .. code::                                                                        |
+                                                                                  |                                                                                  |
+    syn_eqs = biexp_conductance('s', tau1=2.5*ms, tau2=10*ms, E=0*mV,             |    tau1 = 2.5*ms; tau2 = 10*ms; E = 0*mV                                         |
+                                conductance_name='g_syn')                         |    invpeak = (tau2 / tau1) ** (tau1 / (tau2 - tau1))                             |
+    eqs = ... # remaining code as above                                           |    syn_eqs = Equations('''dg_syn/dt = (invpeak*s - g_syn)/tau1 : siemens         |
+                                                                                  |                           ds/dt = -s/tau2 : siemens''')                          |
+                                                                                  |    eqs = ... # remaining code as above                                           |
+                                                                                  |                                                                                  |
+----------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
