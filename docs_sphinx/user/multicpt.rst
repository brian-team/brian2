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
retrieve only the primary branch of this subtree, use the ``branch()`` method::

    mainbranch = morpho.L.branch()

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

    morpho.axon = Morphology(n=5)
    morpho.axon.diameter = ones(5) * 1 * um
    morpho.axon.length = [1 * um, 2 * um, 1 * um, 3 * um, 1 * um]
    morpho.axon.set_coordinates()
    morpho.axon.set_area()

Note the last two statements: ``set_coordinates()`` creates x-y-z coordinates and is required for plotting;
``set_area()`` calculates the area of each compartment (considered as a cylinder)
and is required for using the morphology in simulations.
Alternatively the coordinates can be specified, instead of the lengths of compartments, and then
``set_length()`` must be called. Note that these methods only apply to the main branch of the morphology,
not the children (subtrees).
