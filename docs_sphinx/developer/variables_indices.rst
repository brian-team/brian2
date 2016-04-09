Variables and indices
=====================

Introduction
------------
To be able to generate the proper code out of abstract code statements, the code
generation process has to have access to information about the variables (their
type, size, etc.) as well as to the indices that should be used for indexing
arrays (e.g. a state variable of a `NeuronGroup` will be indexed differently in
the `NeuronGroup` state updater and in synaptic propagation code). Most of this
information is stored in the `variables` attribute of a `VariableOwner` (this
includes `NeuronGroup`, `Synapses`, `PoissonGroup` and everything else that has
state variables). The `variables` attribute can be accessed as a (read-only)
dictionary, mapping variable names to `Variable` objects storing the
information about the respective variable. However, it is not a simple
dictionary but an instance of the `Variables` class. Let's have a look at its
content for a simple example::

    >>> tau = 10*ms
    >>> G = NeuronGroup(10, 'dv/dt = -v / tau : volt')
    >>> for name, var in G.variables.items():
    ...     print('%r : %s' % (name, var))
    ...
   '_spikespace' : <ArrayVariable(unit=Unit(1),  dtype=<type 'numpy.int32'>, scalar=False, constant=False, read_only=False)>
    'i' : <ArrayVariable(unit=Unit(1),  dtype=<type 'numpy.int32'>, scalar=False, constant=True, read_only=True)>
    'N' : <Constant(unit=Unit(1),  dtype=<type 'numpy.int64'>, scalar=True, constant=True, read_only=True)>
    't' : <ArrayVariable(unit=second,  dtype=<type 'numpy.float64'>, scalar=True, constant=False, read_only=True)>
    'v' : <ArrayVariable(unit=volt,  dtype=<type 'numpy.float64'>, scalar=False, constant=False, read_only=False)>
    'dt' : <ArrayVariable(unit=second,  dtype=<type 'float'>, scalar=True, constant=True, read_only=True)>

The state variable ``v`` we specified for the `NeuronGroup` is represented as an
`ArrayVariable`, all the other variables were added automatically. By
convention, internal names for variables that should not be directly accessed by
the user start with an underscore, in the above example the only variable
of this kind is ``'_spikespace'``, the internal datastructure used to store the
spikes that occured in the current time step. There's another array ``i``, the
neuronal indices (simply an array of integers from 0 to 9), that is used for
string expressions involving neuronal indices. The constant ``N`` represents
the total number of neurons. At the first sight it might be surprising that
``t``, the current time of the clock and ``dt``, its timestep, are
`ArrayVariable` objects as well. This is because those values can change during
a run (for ``t``) or between runs (for ``dt``), and storing them as arrays with
a single value (note the ``scalar=True``) is the easiest way to share this value
-- all code accessing it only needs a reference to the array and can access its
only element.

The information stored in the `Variable` objects is used to do various checks
on the level of the abstract code, i.e. before any programming language code is
generated. Here are some examples of errors that are caught this way::

    >>> G.v = 3*ms  # G.variables['v'].unit is volt
    Traceback (most recent call last):
    ...
    DimensionMismatchError: Incorrect units for setting v, dimensions were (s) (m^2 kg s^-3 A^-1)
    >>> G.N = 5  # G.variables['N'] is read-only
    Traceback (most recent call last):
    ...
    TypeError: Variable N is read-only
    >>> G2 = NeuronGroup(10, 'dv/dt = -v / tau : volt', threshold='v')  #G2.variables['v'].is_bool is False
    Traceback (most recent call last):
    ...
    TypeError: Threshold condition "v" is not a boolean expression

Creating variables
------------------
Each variable that should be accessible as a state variable and/or should be
available for use in abstract code has to be created as a `Variable`. For this,
first a `Variables` container with a reference to the group has to be created,
individual variables can then be added using the various ``add_...`` methods::

    self.variables = Variables(self)
    self.variables.add_array('an_array', unit=volt, size=100)
    self.variables.add_constant('N', unit=Unit(1), value=self._N, dtype=np.int32)
    self.variables.create_clock_variables(self.clock)

As an additional argument, array variables can be specified with a specific
*index* (see `Indices`_ below).

References
----------
For each variable, only one `Variable` object exists even if it is used in
different contexts. Let's consider the following example::

    G = NeuronGroup(5, 'dv/dt = -v / tau : volt')
    subG = G[2:]
    S = Synapses(G, G, on_pre='v+=1*mV')
    S.connect()

All allow an access to the state variable `v` (note the different shapes, these
arise from the different indices used, see below)::

    >>> G.v
    <neurongroup.v: array([ 0.,  0.,  0.,  0.,  0.]) * volt>
    >>> subG.v
    <neurongroup_subgroup.v: array([ 0.,  0.,  0.]) * volt>
    >>> S.v
    <synapses.v: array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]) * volt>

In all of these cases, the `Variables` object stores references to the same
`ArrayVariable` object::

    >>> id(G.variables['v'])
    108610960
    >>> id(subG.variables['v'])
    108610960
    >>> id(S.variables['v'])
    108610960

Such a reference can be added using `Variables.add_reference`, note that the
name used for the reference is not necessarily the same as in the original
group, e.g. in the above example ``S.variables`` also stores references to ``v``
under the names ``v_pre`` and ``v_post``.

Indices
-------
In subgroups and especially in synapses, the transformation of abstract code
into executable code is not straightforward because it can involve variables
from different contexts. Here is a simple example::

    G = NeuronGroup(5, 'dv/dt = -v / tau : volt')
    S = Synapses(G, G, 'w : volt', on_pre='v+=w')

The seemingly trivial operation ``v+=w`` involves the variable ``v`` of the
`NeuronGroup` and the variable ``w`` of the `Synapses` object which have to be
indexed in the appropriate way. Since this statement is executed in the context
of ``S``, the variable indices stored there are relevant::

    >>> S.variables.indices['w']
    '_idx'
    >>> S.variables.indices['v']
    '_postsynaptic_idx'

The index ``_idx`` has a special meaning and always refers to the "natural"
index for a group (e.g. all neurons for a `NeuronGroup`, all synapses for a
`Synapses` object, etc.). All other indices have to refer to existing arrays::

    >>> S.variables['_postsynaptic_idx']
    <DynamicArrayVariable(unit=Unit(1),  dtype=<type 'numpy.int32'>, scalar=False, constant=False, is_bool=False, read_only=False)>

In this case, ``_postsynaptic_idx`` refers to a dynamic array that stores the
postsynaptic targets for each synapse (since it is an array itself, it also has
an index. It is defined for each synapse so its index is ``_idx`` -- in fact
there is currently no support for an additional level of indirection in Brian:
a variable representing an index has to have ``_idx`` as its own index). Using
this index information, the following C++ code (slightly simplified) is
generated:

.. code-block:: c++

    for(int _spiking_synapse_idx=0;
    	_spiking_synapse_idx<_num_spiking_synapses;
    	_spiking_synapse_idx++)
    {
    	const int _idx = _spiking_synapses[_spiking_synapse_idx];
    	const int _postsynaptic_idx = _ptr_array_synapses__synaptic_post[_idx];
    	const double w = _ptr_array_synapses_w[_idx];
    	double v = _ptr_array_neurongroup_v[_postsynaptic_idx];
    	v += w;
    	_ptr_array_neurongroup_v[_postsynaptic_idx] = v;
    }

In this case, the "natural" index ``_idx`` iterates over all the synapses that
received a spike (this is defined in the template) and ``_postsynaptic_idx``
refers to the postsynaptic targets for these synapses. The variables ``w`` and
``v`` are then pulled out of their respective arrays with these indices so that
the statement ``v += w;`` does the right thing.

Getting and setting state variables
-----------------------------------
When a state variable is accessed (e.g. using ``G.v``), the group does not
return a reference to the underlying array itself but instead to a
`VariableView` object. This is because a state variable can be accessed in
different contexts and indexing it with a number/array (e.g. ``obj.v[0]``) or
a string (e.g. ``obj.v['i>3']``) can refer to different values in the underlying
array depending on whether the object is the `NeuronGroup`, a `Subgroup` or
a `Synapses` object.

The ``__setitem__`` and ``__getitem__`` methods in `VariableView` delegate to
`VariableView.set_item` and `VariableView.get_item` respectively (which can also
be called directly under special circumstances). They analyze the arguments (is
the index a number, a slice or a string? Is the target value an array or a string
expression?) and delegate the actual retrieval/setting of the values to a
specific method:

* Getting with a numerical (or slice) index (e.g. ``G.v[0]``): `VariableView.get_with_index_array`
* Getting with a string index (e.g. ``G.v['i>3']``): `VariableView.get_with_expression`
* Setting with a numerical (or slice) index and a numerical target value (e.g.
  ``G.v[5:] = -70*mV``): `VariableView.set_with_index_array`
* Setting with a numerical (or slice) index and a string expression value (e.g.
  ``G.v[5:] = (-70+i)*mV``): `VariableView.set_with_expression`
* Setting with a string index and a string expression value (e.g.
  ``G.v['i>5'] = (-70+i)*mV``): `VariableView.set_with_expression_conditional`

These methods are annotated with the `device_override` decorator and can
therefore be implemented in a different way in certain devices. The standalone
device, for example, overrides the all the getting functions and the setting
with index arrays. Note that for standalone devices, the "setter" methods do
not actually set the values but only note them down for later code generation.

Additional variables and indices
--------------------------------
The variables stored in the ``variables`` attribute of a `VariableOwner` can
be used everywhere (e.g. in the state updater, in the threshold, the reset,
etc.). Objects that depend on these variables, e.g. the `Thresholder` of a
`NeuronGroup` add additional variables, in particular `AuxiliaryVariables` that
are automatically added to the abstract code: a threshold condition ``v > 1``
is converted into the statement ``_cond = v > 1``; to specify the meaning of
the variable ``_cond`` for the code generation stage (in particular, C++ code
generation needs to know the data type) an `AuxiliaryVariable` object is created.

In some rare cases, a specific ``variable_indices`` dictionary is provided
that overrides the indices for variables stored in the ``variables`` attribute.
This is necessary for synapse creation because the meaning of the variables
changes in this context: an expression ``v>0`` does not refer to the ``v``
variable of all the *connected* postsynaptic variables, as it does under other
circumstances in the context of a `Synapses` object, but to the ``v`` variable
of all *possible* targets.
