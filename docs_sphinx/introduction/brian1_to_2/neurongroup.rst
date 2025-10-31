Neural models (Brian 1 --> 2 conversion)
========================================
.. sidebar:: Brian 2 documentation

    For the main documentation about defining neural models, see the document
    :doc:`../../user/models`.

.. contents::
    :local:
    :depth: 1

The syntax for specifying neuron models in a `NeuronGroup` changed in several
details. In general, a string-based syntax (that was already optional in Brian 1)
consistently replaces the use of classes (e.g. ``VariableThreshold``) or
guessing (e.g. which variable does ``threshold=50*mV`` check).

Threshold and Reset
-------------------
String-based thresholds are now the only possible option and replace all the
methods of defining threshold/reset in Brian 1:

+----------------------------------------------------------------------------+----------------------------------------------------------------------------+
| Brian 1                                                                    | Brian 2                                                                    |
+============================================================================+============================================================================+
+ .. code::                                                                  | .. code::                                                                  |
+                                                                            |                                                                            |
+    group = NeuronGroup(N, 'dv/dt = -v / tau : volt',                       |    group = NeuronGroup(N, 'dv/dt = -v / tau : volt',                       |
+                        threshold=-50*mV,                                   |                        threshold='v > -50*mV',                             |
+                        reset=-70*mV)                                       |                        reset='v = -70*mV')                                 |
+                                                                            |                                                                            |
+----------------------------------------------------------------------------+----------------------------------------------------------------------------+
+ .. code::                                                                  | .. code::                                                                  |
+                                                                            |                                                                            |
+    group = NeuronGroup(N, 'dv/dt = -v / tau : volt',                       |    group = NeuronGroup(N, 'dv/dt = -v / tau : volt',                       |
+                        threshold=Threshold(-50*mV, state='v'),             |                        threshold='v > -50*mV',                             |
+                        reset=Reset(-70*mV, state='w'))                     |                        reset='v = -70*mV')                                 |
+                                                                            |                                                                            |
+----------------------------------------------------------------------------+----------------------------------------------------------------------------+
+ .. code::                                                                  | .. code::                                                                  |
+                                                                            |                                                                            |
+    group = NeuronGroup(N, '''dv/dt = -v / tau : volt                       |    group = NeuronGroup(N, '''dv/dt = -v / tau : volt                       |
+                              dvt/dt = -vt / tau : volt                     |                              dvt/dt = -vt / tau : volt                     |
+                              vr : volt''',                                 |                              vr : volt''',                                 |
+                        threshold=VariableThreshold(state='v',              |                        threshold='v > vt',                                 |
+                                                    threshold_state='vt'),  |                        reset='v = vr')                                     |
+                        reset=VariableThreshold(state='v',                  |                                                                            |
+                                                resetvaluestate='vr'))      |                                                                            |
+                                                                            |                                                                            |
+----------------------------------------------------------------------------+----------------------------------------------------------------------------+
+ .. code::                                                                  | .. code::                                                                  |
+                                                                            |                                                                            |
+    group = NeuronGroup(N, 'rate : Hz',                                     |    group = NeuronGroup(N, 'rate : Hz',                                     |
+                        threshold=PoissonThreshold(state='rate'))           |                        threshold='rand()<rate*dt')                         |
+                                                                            |                                                                            |
+----------------------------------------------------------------------------+----------------------------------------------------------------------------+

There's no direct equivalent for the "functional threshold/reset" mechanism from
Brian 1. In simple cases, they can be implemented using the general string
expression/statement mechanism (note that in Brian 1, ``reset=myreset`` is
equivalent to ``reset=FunReset(myreset)``):

+------------------------------------------------------------------+-----------------------------------------------------------------+
| Brian 1                                                          | Brian 2                                                         |
+==================================================================+=================================================================+
+ .. code::                                                        | .. code::                                                       |
+                                                                  |                                                                 |
+    def myreset(P,spikes):                                        |    group = NeuronGroup(N, 'dv/dt = -v / tau : volt',            |
+        P.v_[spikes] = -70*mV+rand(len(spikes))*5*mV              |                        threshold='v > -50*mV',                  |
+                                                                  |                        reset='-70*mV + rand()*5*mV')            |
+    group = NeuronGroup(N, 'dv/dt = -v / tau : volt',             |                                                                 |
+                        threshold=-50*mV,                         |                                                                 |
+                        reset=myreset)                            |                                                                 |
+                                                                  |                                                                 |
+------------------------------------------------------------------+-----------------------------------------------------------------+
+ .. code::                                                        | .. code::                                                       |
+                                                                  |                                                                 |
+    def mythreshold(v):                                           |    group = NeuronGroup(N, 'dv/dt = -v / tau : volt',            |
+        return (v > -50*mV) & (rand(N) > 0.5)                     |                        threshold='v > -50*mV and rand() > 0.5', |
+                                                                  |                        reset='v = -70*mV')                      |
+    group = NeuronGroup(N, 'dv/dt = -v / tau : volt',             |                                                                 |
+                        threshold=SimpleFunThreshold(mythreshold, |                                                                 |
+                                                     state='v'),  |                                                                 |
+                        reset=-70*mV)                             |                                                                 |
+                                                                  |                                                                 |
+------------------------------------------------------------------+-----------------------------------------------------------------+

For more complicated cases, you can use the general mechanism for
:ref:`user_functions` that Brian 2 provides. The only caveat is that you'd have
to provide an implementation of the function in the code generation target
language which is by default C++ or Cython. However, in the default
:ref:`runtime` mode, you can chose different code generation targets for
different parts of your simulation. You can thus switch the code generation
target for the threshold/reset mechanism to ``numpy`` while leaving the default
target for the rest of the simulation in place. The details of this process and
the correct definition of the functions (e.g. ``global_reset`` needs a "dummy"
return value) are somewhat cumbersome at the moment and we plan to make them
more straightforward in the future. Also note that if you use this kind of
mechanism extensively, you'll lose all the performance advantage that Brian 2's
code generation mechanism provides (in addition to not being able to use
:ref:`cpp_standalone` mode at all).

+-------------------------------------------------------------------------+-----------------------------------------------------------------+
| Brian 1                                                                 | Brian 2                                                         |
+=========================================================================+=================================================================+
+ .. code::                                                               | .. code::                                                       |
+                                                                         |                                                                 |
+    def single_threshold(v):                                             |    @check_units(v=volt, result=bool)                            |
+        # Only let a single neuron spike                                 |    def single_threshold(v):                                     |
+        crossed_threshold = np.nonzero(v > -50*mV)[0]                    |        pass # ... (identical to Brian 1)                        |
+        should_spike = np.zeros(len(P), dtype=np.bool)                   |                                                                 |
+        if len(crossed_threshold):                                       |    @check_units(spikes=1, result=1)                             |
+            choose = np.random.randint(len(crossed_threshold))           |    def global_reset(spikes):                                    |
+            should_spike[crossed_threshold[choose]] = True               |        # Reset everything                                       |
+        return should_spike                                              |        if len(spikes):                                          |
+                                                                         |             neurons.v_[:] = -0.070                              |
+    def global_reset(P, spikes):                                         |                                                                 |
+        # Reset everything                                               |    neurons = NeuronGroup(N, 'dv/dt = -v / tau : volt',          |
+        if len(spikes):                                                  |                          threshold='single_threshold(v)',       |
+            P.v_[:] = -70*mV                                             |                          reset='dummy = global_reset(i)')       |
+                                                                         |    # Set the code generation target for threshold/reset only:   |
+    neurons = NeuronGroup(N, 'dv/dt = -v / tau : volt',                  |    neuron.thresholder['spike'].codeobj_class = NumpyCodeObject  |
+                          threshold=SimpleFunThreshold(single_threshold, |    neuron.resetter['spike'].codeobj_class = NumpyCodeObject     |
+                                                       state='v'),       |                                                                 |
+                          reset=global_reset)                            |                                                                 |
+                                                                         |                                                                 |
+-------------------------------------------------------------------------+-----------------------------------------------------------------+

For an example how to translate ``EmpiricalThreshold``, see the section on
"Refractoriness" below.

Refractoriness
--------------
For a detailed description of Brian 2's refractoriness mechanism see
:doc:`../../user/refractoriness`.

In Brian 1, refractoriness was tightly linked with the reset mechanism and
some combinations of refractoriness and reset were not allowed. The standard
refractory mechanism had two effects during the refractoriness: it prevented the
refractory cell from spiking and it clamped a state variable (normally the
membrane potential of the cell). In Brian 2, refractoriness is independent of
reset and the two effects are specified separately: the ``refractory`` keyword
specifies the time (or an expression evaluating to a time) during which the
cell does not spike, and the ``(unless refractory)`` flag marks one or more
variables to be clamped during the refractory period. To correctly translate
the standard refractory mechanism from Brian 1, you'll therefore need to
specify both:

+---------------------------------------------------------+-----------------------------------------------------------------------------+
| Brian 1                                                 | Brian 2                                                                     |
+=========================================================+=============================================================================+
+ .. code::                                               | .. code::                                                                   |
+                                                         |                                                                             |
+    group = NeuronGroup(N, 'dv/dt = (I - v)/tau : volt', |    group = NeuronGroup(N, 'dv/dt = (I - v)/tau : volt (unless refractory)', |
+                        threshold=-50*mV,                |                        threshold='v > -50*mV',                              |
+                        reset=-70*mV,                    |                        reset='v = -70*mV',                                  |
+                        refractory=3*ms)                 |                        refractory=3*ms)                                     |
+                                                         |                                                                             |
+---------------------------------------------------------+-----------------------------------------------------------------------------+

More complex refractoriness mechanisms based on ``SimpleCustomRefractoriness``
and ``CustomRefractoriness`` can be translatated using string expressions or
user-defined functions, see the remarks in the preceding section on "Threshold
and Reset".

Brian 2 no longer has an equivalent to the ``EmpiricalThreshold`` class (which
detects at the first threshold crossing but ignores all following threshold
crossings for a certain time after that). However, the standard refractoriness
mechanism can be used to implement the same behaviour, since it does not
reset/clamp any value if not explicitly asked for it (which would be fatal for
Hodgkin-Huxley type models):

+----------------------------------------------------------------------+----------------------------------------------------------------------+
| Brian 1                                                              | Brian 2                                                              |
+======================================================================+======================================================================+
+ .. code::                                                            | .. code::                                                            |
+                                                                      |                                                                      |
+    group = NeuronGroup(N,'''                                         |    group = NeuronGroup(N,'''                                         |
+                        dv/dt = (I_L - I_Na - I_K + I)/Cm : volt      |                        dv/dt = (I_L - I_Na - I_K + I)/Cm : volt      |
+                        ...''',                                       |                        ...''',                                       |
+                        threshold=EmpiricalThreshold(threshold=20*mV, |                        threshold='v > -20*mV',                       |
+                                                     refractory=1*ms, |                        refractory=1*ms)                              |
+                                                     state='v'))      |                                                                      |
+                                                                      |                                                                      |
+----------------------------------------------------------------------+----------------------------------------------------------------------+

Subgroups
---------
The class `NeuronGroup` in Brian 2 does no longer provide a ``subgroup`` method,
the only way to construct subgroups is therefore the slicing syntax (that works
in the same way as in Brian 1):

+-------------------------------------+-----------------------------------+
| Brian 1                             | Brian 2                           |
+=====================================+===================================+
+ .. code::                           | .. code::                         |
+                                     |                                   |
+    group = NeuronGroup(4000, ...)   |    group = NeuronGroup(4000, ...) |
+    group_exc = group.subgroup(3200) |    group_exc = group[:3200]       |
+    group_inh = group.subgroup(800)  |    group_inh = group[3200:]       |
+                                     |                                   |
+-------------------------------------+-----------------------------------+

Linked Variables
----------------
For a description of Brian 2's mechanism to link variables between groups, see
:ref:`linked_variables`.

Linked variables need to be explicitly annotated with the ``(linked)`` flag in
Brian 2:

+----------------------------------------------------------+----------------------------------------------------------+
| Brian 1                                                  | Brian 2                                                  |
+==========================================================+==========================================================+
+ .. code::                                                | .. code::                                                |
+                                                          |                                                          |
+    group1 = NeuronGroup(N,                               |    group1 = NeuronGroup(N,                               |
+                         'dv/dt = -v / tau : volt')       |                         'dv/dt = -v / tau : volt')       |
+    group2 = NeuronGroup(N,                               |    group2 = NeuronGroup(N,                               |
+                         '''dv/dt = (-v + w) / tau : volt |                         '''dv/dt = (-v + w) / tau : volt |
+                            w : volt''')                  |                            w : volt (linked)''')         |
+    group2.w = linked_var(group1, 'v')                    |    group2.w = linked_var(group1, 'v')                    |
+                                                          |                                                          |
+----------------------------------------------------------+----------------------------------------------------------+
