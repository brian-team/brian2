Multicompartmental models (Brian 1 --> 2 conversion)
====================================================
.. sidebar:: Brian 2 documentation

    Support for multicompartmental models is now an integral part of Brian 2
    (an early version of it was included as an experimental module in Brian 1).
    See the document :doc:`../../user/multicompartmental`.

Brian 1 offered support for simple multi-compartmental models in the
``compartments`` module. This module allowed you to combine the equations for
several compartments into a single `Equations` object. This is only a suitable
solution for simple morphologies (e.g. "ball-and-stick" models) but has the
advantage over using `SpatialNeuron` that you can have several of such neurons
in a `NeuronGroup`.

If you already have a definition of a model using Brian 1's ``compartments``
module, then you can simply print out the equations and use them directly in
Brian 2. For simple models, writing the equations without that help is rather
straightforward anyway:

+---------------------------------------------------+---------------------------------------------------+
| Brian 1                                           | Brian 2                                           |
+===================================================+===================================================+
| .. code::                                         | .. code::                                         |
|                                                   |                                                   |
|    V0 = 10*mV                                     |    V0 = 10*mV                                     |
|    C = 200*pF                                     |    C = 200*pF                                     |
|    Ra = 150*kohm                                  |    Ra = 150*kohm                                  |
|    R = 50*Mohm                                    |    R = 50*Mohm                                    |
|    soma_eqs = (MembraneEquation(C) +              |    neuron_eqs = '''                               |
|                IonicCurrent('I=(vm-V0)/R : amp')) |    dvm_soma/dt = (I_soma + I_soma_dend)/C : volt  |
|    dend_eqs = MembraneEquation(C)                 |    I_soma = (V0 - vm_soma)/R : amp                |
|    neuron_eqs = Compartments({'soma': soma_eqs,   |    I_soma_dend = (vm_dend - vm_soma)/Ra : amp     |
|                               'dend': dend_eqs})  |    dvm_dend/dt = -I_soma_dend/C : volt'''         |
|                                                   |                                                   |
|    neuron = NeuronGroup(N, neuron_eqs)            |    neuron = NeuronGroup(N, neuron_eqs)            |
|                                                   |                                                   |
+---------------------------------------------------+---------------------------------------------------+
