Multicompartment models
=======================

It is possible to create neuron models with a spatially extended morphology, using
the `SpatialNeuron` class. A `SpatialNeuron` is a single neuron with many compartments.
Essentially, it works as a `NeuronGroup` where elements are compartments instead of neurons.
A `SpatialNeuron` is specified mainly by a set of equations for transmembrane currents (ionic channels)
and a morphology.

Creating a neuron morphology
----------------------------

Schematic morphologies
~~~~~~~~~~~~~~~~~~~~~~
Morphologies can be created combining different standard geometrical objects::

    soma = Soma(diameter=30*um)
    cylinder = Cylinder(10, length=100*um, diameter=1*um)
    section = Section(5, length=100*um, diameter=[5, 4, 3, 2, 1])*um

The first statement creates a single iso-potential compartment (i.e. with no axial resistance within the compartment),
with its area calculated as the area of a sphere with the given diameter. The second one specifies a cylinder consisting
of 10 compartments with identical diameter and the given total length. Finally, the third statement creates a cable of
5 compartments with the given total length, where each compartment is a truncated cone with the given diameters. Note
that here, only the diameter at the end of each compartment was specified, which means that the diameter at the start
of the first compartment will be automatically taken as the diameter of the parent compartment.

The following table summarizes the different options to create schematic morphologies:

+-------------+-----------------------------------------------------------------------------------+
|             | **Example**                                                                       |
+=============+===================================================================================+
|**Soma**     |  ::                                                                               |
|             |                                                                                   |
|             |      # Soma always has a single compartment                                       |
|             |      Soma(diameter=30*um)                                                         |
|             |                                                                                   |
|             | .. image:: images/soma.*                                                          |
|             |                                                                                   |
+-------------+-----------------------------------------------------------------------------------+
|**Cylinder** |  ::                                                                               |
|             |                                                                                   |
|             |     # Defining total length                                                       |
|             |     Cylinder(5, diameter=10*um, length=50*um)                                     |
|             |                                                                                   |
|             | .. image:: images/cylinder_1.*                                                    |
|             |                                                                                   |
+-------------+-----------------------------------------------------------------------------------+
|             |  ::                                                                               |
|             |                                                                                   |
|             |     # Defining lengths of individual compartments                                 |
|             |     Cylinder(5, diameter=10*um, length=[10, 20, 5, 5, 10]*um)                     |
|             |                                                                                   |
|             | .. image:: images/cylinder_2.*                                                    |
|             |                                                                                   |
+-------------+-----------------------------------------------------------------------------------+
|             |  ::                                                                               |
|             |                                                                                   |
|             |     # Diameters of individual compartments and total length                       |
|             |     Cylinder(5, diameter=[5, 10, 5, 10, 5]*um, length=50*um)                      |
|             |                                                                                   |
|             | .. image:: images/cylinder_3.*                                                    |
|             |                                                                                   |
+-------------+-----------------------------------------------------------------------------------+
|             |  ::                                                                               |
|             |                                                                                   |
|             |     # Lengths and diameters of individual compartments                            |
|             |     Cylinder(5, diameter=[5, 10, 5, 10, 5]*um, length=[10, 20, 5, 5, 10]*um)      |
|             |                                                                                   |
|             | .. image:: images/cylinder_4.*                                                    |
|             |                                                                                   |
+-------------+-----------------------------------------------------------------------------------+
|**Section**  |  ::                                                                               |
|             |                                                                                   |
|             |     # Total length, constant diameter (first diameter is diameter of parent)      |
|             |     Section(5, diameter=10*um, length=50*um)                                      |
|             |                                                                                   |
|             | .. image:: images/section_1.*                                                     |
|             |                                                                                   |
+-------------+-----------------------------------------------------------------------------------+
|             |  ::                                                                               |
|             |                                                                                   |
|             |     # Lengths of individual compartments (first diameter is diameter of parent)   |
|             |     Section(5, diameter=10*um, length=[10, 20, 5, 5, 10]*um)                      |
|             |                                                                                   |
|             | .. image:: images/section_2.*                                                     |
|             |                                                                                   |
+-------------+-----------------------------------------------------------------------------------+
|             |  ::                                                                               |
|             |                                                                                   |
|             |     # Total length and diameters of individual compartments                       |
|             |     # (diameters at the end of compartments, first diameter is parent diameter)   |
|             |     Section(5, diameter=[5, 10, 5, 10, 5]*um, length=50*um)                       |
|             |                                                                                   |
|             | .. image:: images/section_3.*                                                     |
|             |                                                                                   |
+-------------+-----------------------------------------------------------------------------------+
|             |  ::                                                                               |
|             |                                                                                   |
|             |     # Lengths and diameters of individual compartments                            |
|             |     # (diameters at the end of compartments, first diameter is parent diameter)   |
|             |     Section(5, diameter=[5, 10, 5, 10, 5]*um, length=[10, 20, 5, 5, 10]*um)       |
|             |                                                                                   |
|             | .. image:: images/section_4.*                                                     |
|             |                                                                                   |
+-------------+-----------------------------------------------------------------------------------+
|             |  ::                                                                               |
|             |                                                                                   |
|             |     # Total length and diameters of individual compartments, including start      |
|             |     # diameter                                                                    |
|             |     Section(5, diameter=[2.5, 5, 10, 5, 10, 5]*um, length=50*um)                  |
|             |                                                                                   |
|             | .. image:: images/section_5.*                                                     |
|             |                                                                                   |
+-------------+-----------------------------------------------------------------------------------+
|             |  ::                                                                               |
|             |                                                                                   |
|             |     # Lengths and diameters of individual compartments, including start           |
|             |     # diameter                                                                    |
|             |     Section(5, diameter=[2.5, 5, 10, 5, 10, 5]*um, length=[10, 20, 5, 5, 10]*um)  |
|             |                                                                                   |
|             | .. image:: images/section_6.*                                                     |
|             |                                                                                   |
+-------------+-----------------------------------------------------------------------------------+


The tree structure of a morphology is created by attaching `Morphology` objects together::

    morpho = Soma(diameter=30*um)
    morpho.axon = Cylinder(length=100*um, diameter=1*um, n=10)
    morpho.dendrite = Cylinder(length=50*um, diameter=2*um, n=5)

These statements create a morphology consisting of a cylindrical axon and a dendrite attached to a spherical soma.
Note that the names ``axon`` and ``dendrite`` are arbitrary and chosen by the user. For example, the same morphology can
be created as follows::

    morpho = Soma(diameter = 30*um)
    morpho.output_process = Cylinder(length=100*um, diameter=1*um, n=10)
    morpho.input_process = Cylinder(length=50*um, diameter=2*um, n=5)

The syntax is recursive, for example two branches can be added at the end of the dendrite as follows::

    morpho.dendrite.branch1 = Cylinder(length=50*um, diameter=1*um, n=3)
    morpho.dendrite.branch2 = Cylinder(length=50*um, diameter=1*um, n=3)

Equivalently, one can use an indexing syntax::

    morpho['dendrite']['branch1'] = Cylinder(length=50*um, diameter=1*um, n=3)
    morpho['dendrite']['branch2'] = Cylinder(length=50*um, diameter=1*um, n=3)

Finally there is a special shorter syntax for quickly creating trees, using ``L`` (for left),
``R`` (for right), and digits from 1 to 9. These can be simply concatenated (without using the dot)::

    morpho.L = Cylinder(length=10*um, diameter=1*um, n=3)
    morpho.L1 = Cylinder(length=5*um, diameter=1*um, n=3)
    morpho.L2 = Cylinder(length=5*um, diameter=1*um, n=3)
    morpho.L3 = Cylinder(length=5*um, diameter=1*um, n=3)
    morpho.R = Cylinder(length=10*um, diameter=1*um, n=3)
    morpho.RL = Cylinder(length=5*um, diameter=1*um, n=3)
    morpho.RR = Cylinder(length=5*um, diameter=1*um, n=3)

These instructions create a dendritic tree with two main branches, 3 subbranches on the first branch and
2 on the second branch. After these instructions are executed, ``morpho.L`` contains the entire subtree. However,
accessing the attributes (e.g. ``diameter``) will only return the values for the given section.

.. note::

    To avoid ambiguities, do not use names for sections that can be interpreted in the abreviated way detailed above.
    For example, do not name a child branch ``L1`` (which will be interpreted as the first child of the child ``L``)


The number of compartments in a section can be accessed with ``morpho.n`` (or ``morpho.L.n``, etc.), the number of
total sections and compartments in a subtree can be accessed with ``morpho.n_sections`` and ``len(morpho)``
respectively.

.. todo::

    Explain ``generate_coordinates``

Complex morphologies
~~~~~~~~~~~~~~~~~~~~

Morphologies can also be created from information about the compartment coordinates in 3D space. This can be done
manually for individual sections, following the same syntax as the "schematic" morphologies::

    soma = Soma(diameter=30*um, x=50*um, y=20*um)
    cylinder = Cylinder(10, x=100*um, diameter=1*um)
    section = Section(5,
                      x=[10, 20, 30, 40, 50]*um,
                      y=[10, 20, 30, 40, 50]*um,
                      z=[10, 10, 10, 10, 10]*um,
                      diameter=[5, 4, 3, 2, 1])*um

A few notes:

1. In the vast majority of simulations, coordinates are not used in the neuronal equations, therefore the
   coordinates are purely for visualization purposes and do not affect the simulation results in any way.
2. Coordinate specification cannot be combined with length specification -- lengths are automatically calculated from
   the coordinates.
3. The coordinate specification can also be 1- or 2-dimensional (as in the first two examples above), the unspecified
   coordinate will be taken from the value of the parent section (or as 0 μm for the root section)
4. Similar to the ``length`` argument, a single argument for multiple compartments (see the `Cylinder` example above) is
   interpreted as the point at the end of the section.
5. All coordinates are interpreted relative to the parent compartment, i.e. the point (0 μm, 0 μm, 0 μm) refers to the
   end point of the previous compartment. There is one exception to this rule: if the section has ``n`` compartments,
   and ``n+1`` coordinate values have been given, then the first point is interpreted as the start point of the section
   and all values are considered to be *absolute*. This is similar to the semantics of the ``diameter`` argument of
   `Section` and is mostly useful for morphologies created from neuronal reconstructions (see below). Another use is the
   connection of dendrites and axons to a soma, which otherwise will be connected to the center of the sphere (as noted
   before, this is only relevant for visualization).

A neuronal morphology can be directly load from a ``.swc`` file (a standard format for neuronal morphologies)::

    morpho = Morphology.from_file('corticalcell.swc')

There is a large database of morphologies in the swc format at http://neuromorpho.org.

To manually create a morphology from a list of points in a similar format to SWC files, see `Morphology.from_points`

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
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1*uF/cm**2, Ri=100*ohm*cm)
    neuron.v = EL+10*mV

Several state variables are created automatically: the `SpatialNeuron` inherits all the geometrical variables of the
compartments (``length``, ``diameter``, ``area``, ``volume``), as well as the ``distance`` variable that gives the
distance to the soma. For morphologies that use coordinates, the ``x``, ``y`` and ``z`` variables are provided as well.
Additionally, a state variable ``Cm`` is created. It is initialized with the value given at construction, but it can be
modified on a compartment per compartment basis (which is useful to model myelinated axons). The membrane potential is
stored in state variable ``v``.

Note that for all variable values that vary across a compartment (e.g. ``distance``, ``x``, ``y``, ``z``, ``v``), the
value that is reported is the value at the "electrical midpoint" (the point with identical axial resistance to the two
ends) of the compartment. For spherical and cylindrical compartments, this midpoints simply corresponds to the
geometrical midpoint, but for compartments modeled as truncated cones with different diameters at their start and end,
the electrical midpoint is closer to the end with the bigger diameter.

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

Note that the state variables correspond to the entire subtree, not just the main section.
That is, if the axon had branches, then the above statement would change ``gNa`` on the main section
and all the sections in the subtree. To access the main section only, use the attribute ``main``::

    neuron.axon.main.gNa = 10*gL

A typical use case is when one wants to change parameter values at the soma only. For example, inserting
an electrode current at the soma is done as follows::

    neuron.main.I = 1*nA

A part of a branch can be accessed as follows::

    initial_segment = neuron.axon[10*um:50*um]

Synaptic inputs
~~~~~~~~~~~~~~~
There are two methods to have synapses on `SpatialNeuron`.
The first one to insert synaptic equations directly in the neuron equations::

    eqs='''
    Im = gL*(EL-v) : amp/meter**2
    Is = gs*(Es-v) : amp (point current)
    dgs/dt = -gs/taus : siemens
    '''
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1*uF/cm**2, Ri=100*ohm*cm)

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
~~~~~~~~~~~~~~~~
To detect and record spikes, we must specify a threshold condition, essentially in the same
way as for a `NeuronGroup`::

    neuron = SpatialNeuron(morphology=morpho, model=eqs, threshold='v > 0*mV', refractory='v > -10*mV')

Here spikes are detected when the membrane potential ``v`` reaches 0 mV. Because there is generally
no explicit reset in this type of model (although it is possible to specify one), ``v`` remains above
0 mV for some time. To avoid detecting spikes during this entire time, we specify a refractory period.
In this case no spike is detected as long as ``v`` is greater than -10 mV. Another possibility could be::

    neuron = SpatialNeuron(morphology=morpho, model=eqs, threshold='m > 0.5', refractory='m > 0.4')

where ``m`` is the state variable for sodium channel activation (assuming this has been defined in the
model). Here a spike is detected when half of the sodium channels are open.

With the syntax above, spikes are detected in all compartments of the neuron. To detect them in a single
compartment, use the ``threshold_location`` keyword::

    neuron = SpatialNeuron(morphology=morpho, model=eqs, threshold='m > 0.5', threshold_location=30,
                           refractory='m > 0.4')

In this case, spikes are only detecting in compartment number 30. Reset then applies locally to
that compartment (if a reset statement is defined).
Again the location of the threshold can be specified with spatial position::

    neuron = SpatialNeuron(morphology=morpho, model=eqs, threshold='m > 0.5',
                           threshold_location=morpho.axon[30*um],
                           refractory='m > 0.4')
