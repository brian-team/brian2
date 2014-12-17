Multicompartment models
=======================

It is possible to create neuron models with a spatially extended morphology, using
the `SpatialNeuron` class. A `SpatialNeuron` is a single neuron with many compartments.
Essentially, it works as a `NeuronGroup` where elements are compartments instead of neurons.
A `SpatialNeuron` is specified mainly by a set of equations for transmembrane currents (ionic channels)
and a morphology.

Creating a neuron morphology
----------------------------
Morphologies are stored and manipulated with
`Morphology` objects. A new morphology can be loaded from a ``.swc`` file (a standard format for neuronal morphologies)::

    morpho = Morphology('corticalcell.swc')

There is a large database of morphologies in the swc format at http://neuromorpho.org/neuroMorpho.

Morphologies can also be created manually by combining different standard geometrical objects::

    morpho = Soma(diameter = 30*um)
    morpho = Cylinder(length = 100*um, diameter = 1*um, n = 10)

The first statement creates a sphere (as a single compartment), the second one a cylinder with 10 compartments
(default is 1 compartment). It is also possible to specify the type of process,
which is ``soma``, ``axon`` or ``dendrite``, and the relative coordinates of the end point. For example::

    morpho = Cylinder(diameter = 1*um, n = 10, type = 'axon', x = 50*um, y = 100*um, z = 0*um)

In this case, length must not be specified, as it is calculated from the coordinates.
These coordinates are mostly helpful for visualization. If they are not specified, 3D direction is chosen at
random.

A tree is created by attaching `Morphology` objects together::

    morpho = Soma(diameter = 30*um)
    morpho.axon = Cylinder(length = 100*um, diameter = 1*um, n = 10)
    morpho.dendrite = Cylinder(length = 50*um, diameter = 2*um, n = 5)

These statements create a morphology consisting of a cylindrical axon and a dendrite attached to a spherical soma.
Note that the names ``axon`` and ``dendrite`` are arbitrary and chosen by the user. For example, the same morphology can
be created as follows::

    morpho = Soma(diameter = 30*um)
    morpho.output_process = Cylinder(length = 100*um, diameter = 1*um, n = 10)
    morpho.input_process = Cylinder(length = 50*um, diameter = 2*um, n = 5)

The syntax is recursive, for example two branches can be added at the end of the dendrite as follows::

    morpho.dendrite.branch1 = Cylinder(length = 50*um, diameter = 1*um, n = 3)
    morpho.dendrite.branch2 = Cylinder(length = 50*um, diameter = 1*um, n = 3)

Equivalently, one can use an indexing syntax::

    morpho['dendrite']['branch1'] = Cylinder(length = 50*um, diameter = 1*um, n = 3)
    morpho['dendrite']['branch2'] = Cylinder(length = 50*um, diameter = 1*um, n = 3)

Finally there is a special shorter syntax for quickly creating trees, using ``L`` (for left),
``R`` (for right), and digits from 1 to 9. These can be simply concatenated (without using the dot)::

    morpho.L=Cylinder(length = 10*um, diameter = 1 *um, n = 3)
    morpho.L1=Cylinder(length = 5*um, diameter = 1 *um, n = 3)
    morpho.L2=Cylinder(length = 5*um, diameter = 1 *um, n = 3)
    morpho.L3=Cylinder(length = 5*um, diameter = 1 *um, n = 3)
    morpho.R=Cylinder(length = 10*um, diameter = 1 *um, n = 3)
    morpho.RL=Cylinder(length = 5*um, diameter = 1 *um, n = 3)
    morpho.RR=Cylinder(length = 5*um, diameter = 1 *um, n = 3)

These instructions create a dendritic tree with two main branches, 3 subbranches on the first branch and
2 on the second branch. After these instructions are executed, ``morpho.L`` contains the entire subtree. To
retrieve only the primary branch of this subtree, use the ``main`` attribute::

    mainbranch = morpho.L.main

The number of compartments in the entire tree can be obtained with
``len(morpho)``. Finally, the morphology can be displayed as a 3D plot::

    morpho.plot()

Complex morphologies
~~~~~~~~~~~~~~~~~~~~
Neuronal morphologies can be created by assembling cylinders and spheres, but also more complex processes with
variable diameter. This can be done by directly setting the attributes of a `Morphology` object.
A `Morphology` object stores the ``diameter``, ``x``, ``y``, ``z``, ``length`` and ``area`` of all the
compartments of the main branch (i.e., not children) as arrays. For example, ``morpho.length`` is
an array containing the length of each of its compartments. When creating
a cylinder, the length of each compartment is obtained by dividing the total length provided at creation by the
number of compartments. The area is calculated automatically.
Complex processes can be created manually by directly specifying the diameter and length of
each compartment::

    morpho.axon = Morphology(n = 5)
    morpho.axon.diameter = ones(5) * 1 * um
    morpho.axon.length = [1, 2, 1, 3, 1] * um
    morpho.axon.set_coordinates()
    morpho.axon.set_area()

Note the last two statements: ``set_coordinates()`` creates x-y-z coordinates and is required for plotting;
``set_area()`` calculates the area of each compartment (considered as a cylinder)
and is required for using the morphology in simulations.
Alternatively the coordinates can be specified, instead of the lengths of compartments, and then
``set_length()`` must be called. Note that these methods only apply to the main branch of the morphology,
not the children (subtrees).

Creating a spatially extended neuron
------------------------------------

A `SpatialNeuron` is a spatially extended neuron. It is created by specifying the morphology as a
`Morphology` object, the equations for transmembrane currents, and optionally the specific membrane capacitance
``Cm`` and intracellular resistivity ``Ri``::

    gL=1e-4*siemens/cm**2
    EL=-70*mV
    eqs='''
    Im=gL*(EL-v) : amp/meter**2
    I : amp (point current)
    '''
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
    neuron.v = EL+10*mV

Several state variables are created automatically: all the variables of the morphology object are linked to
state variables of the neuron (``diameter``, ``x``, ``y``, ``z``, ``length`` and ``area``). Additionally,
a state variable ``Cm`` is created. It is initialized with the value given at construction, but it can be modified
on a compartment per compartment basis (which is useful to model myelinated axons).
Finally the membrane potential is stored in state variable ``v``.
The integration method can be specified as for a `NeuronGroup` with the ``method`` keyword.
In general, for models with nonlinear conductances, the exponential Euler method should be used:
``method = "exponential_euler"``.

The key state variable, which must be specified at construction, is ``Im``. It is the total transmembrane current,
expressed in units of current per area. This is a mandatory line in the definition of the model. The rest of the
string description may include other state variables (differential equations or subexpressions)
or parameters, exactly as in `NeuronGroup`. At every timestep, Brian integrates the state variables, calculates the
transmembrane current at every point on the neuronal morphology, and updates ``v`` using the transmembrane current and
the diffusion current, which is calculated based on the morphology and the intracellular resistivity.
Note that the transmembrane current is a surfacic current, not the total current in the compartement.
This choice means that the model equations are independent of the number of compartments chosen for the simulation.
The space constant can obtained for any point of the neuron with the ``space_constant`` attribute::

    l = neuron.space_constant[0]

The calculation is based on the local total conductance (not just the leak conductance).
Therefore, it can potentially vary during a simulation (e.g. decrease during an action potential).

To inject a current `I` at a particular point (e.g. through an electrode or a synapse), this current must be divided by
the area of the compartment when inserted in the transmembrane current equation. This is done automatically when
the flag ``point current`` is specified, as in the example above. This flag can apply only to subexpressions or
parameters with amp units. Internally, the expression of the transmembrane current ``Im`` is simply augmented with
``+I/area``. A current can then be injected in the first compartment of the neuron (generally the soma) as follows::

    neuron.I[0]=1*nA

State variables of the `SpatialNeuron` include all the compartments of that neuron (including subtrees).
Therefore, the statement ``neuron.v=EL+10*mV`` sets the membrane potential of the entire neuron at -60 mV.

Subtrees can be accessed by attribute (in the same way as in `Morphology` objects)::

    neuron.axon.gNa = 10*gL

Note that the state variables correspond to the entire subtree, not just the main branch.
That is, if the axon had branches, then the above statement would change ``gNa`` on the main branch
and all the subbranches. To access the main branch only, use the attribute ``main``::

    neuron.axon.main.gNa = 10*gL

A typical use case is when one wants to change parameter values at the soma only. For example, inserting
an electrode current at the soma is done as follows::

    neuron.main.I = 1*nA

A part of a branch can be accessed as follows::

    initial_segment = neuron.axon[10*um:50*um]

Synaptic inputs
---------------
There are two methods to have synapses on `SpatialNeuron`.
The first one to insert synaptic equations directly in the neuron equations::

    eqs='''
    Im = gL*(EL-v) : amp/meter**2
    Is = gs*(Es-v) : amp (point current)
    dgs/dt = -gs/taus : siemens
    '''
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)

Note that, as for electrode stimulation, the synaptic current must be defined as a point current.
Then we use a `Synapses` object to connect a spike source to the neuron::

    S = Synapses(stimulation,neuron,pre = 'gs += w')
    S.connect(0,50)
    S.connect(1,100)

This creates two synapses, on compartments 50 and 100. One can specify the compartment number
with its spatial position by indexing the morphology::

    S.connect(0,morpho[25*um])
    S.connect(1,morpho.axon[30*um])

In this method for creating synapses,
there is a single value for the synaptic conductance in any compartment.
This means that it will fail if there are several synapses onto the same compartment and synaptic equations
are nonlinear.
The second method, which works in such cases, is to have synaptic equations in the
`Synapses` object::

    eqs='''
    Im = gL*(EL-v) : amp/meter**2
    Is = gs*(Es-v) : amp (point current)
    gs : siemens
    '''
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
    S = Synapses(stimulation,neuron,model='''dg/dt = -g/taus : siemens
                                             gs_post = g : siemens (summed)''',pre = 'g += w')

Here each synapse (instead of each compartment) has an associated value ``g``, and all values of
``g`` for each compartment (i.e., all synapses targeting that compartment) are collected
into the compartmental variable ``gs``.

Detecting spikes
----------------
To detect and record spikes, we must specify a threshold condition, essentially in the same
way as for a `NeuronGroup`::

    neuron = SpatialNeuron(morphology=morpho, model=eqs, threshold = "v > 0*mV", refractory = "v > -10*mV")

Here spikes are detected when the membrane potential ``v`` reaches 0 mV. Because there is generally
no explicit reset in this type of model (although it is possible to specify one), ``v`` remains above
0 mV for some time. To avoid detecting spikes during this entire time, we specify a refractory period.
In this case no spike is detected as long as ``v`` is greater than -10 mV. Another possibility could be::

    neuron = SpatialNeuron(morphology=morpho, model=eqs, threshold = "m > 0.5", refractory = "m > 0.4")

where ``m`` is the state variable for sodium channel activation (assuming this has been defined in the
model). Here a spike is detected when half of the sodium channels are open.

With the syntax above, spikes are detected in all compartments of the neuron. To detect them in a single
compartment, use the ``threshold_location`` keyword::

    neuron = SpatialNeuron(morphology=morpho, model=eqs, threshold = "m > 0.5", threshold_location = 30,
                           refractory = "m > 0.4")

In this case, spikes are only detecting in compartment number 30. Reset then applies locally to
that compartment (if a reset statement is defined).
Again the location of the threshold can be specified with spatial position::

    neuron = SpatialNeuron(morphology=morpho, model=eqs, threshold = "m > 0.5",
                           threshold_location = morpho.axon[30*um],
                           refractory = "m > 0.4")
