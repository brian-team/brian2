Standalone implementation
=========================

.. contents::
    :local:
    :depth: 1

This – currently very incomplete – document describes some of the implementation details of :ref:`cpp_standalone`.

.. _array_cache:
Array cache
-----------
As described in :ref:`standalone variables <standalone_variables>`, in standalone mode Python code does not
usually have access to state variables and synaptic indices, since the code necessary to initialize/create them has
not been run yet. Concretely, accessing a state variable (or other variables like synaptic indices), will call
`.ArrayVariable.get_value` which delegates to `.CPPStandaloneDevice.get_value`. After a run, this will read the
corresponding file from the disk and return the values. The user can therefore use the same code to analyze the
results as for runtime mode. Before a run, this file does not exist, but `.CPPStandaloneDevice.get_value` has another
mechanism to return values: the "array cache". This cache is a simple dictionary, stored in
`.CPPStandaloneDevice.array_cache`, mapping `.ArrayVariable` objects to their respective values. If the requested
object is present in this cache, its values can be accessed even before the simulation is run. Values are added
to this cache, whenever simulation code sets variables with concrete values. Methods such as
`.CPPStandaloneDevice.fill_with_array` or `.CPPStandaloneDevice.init_with_zeros` write the provided values
into the array cache so that they can be retrieved later. Conversely, `.CPPStandaloneDevice.code_object` will delete
any existing information in ``array_cache`` for variables that are changed by a code object, i.e. invalidate any
previously stored values::

    >>> set_device('cpp_standalone')
    >>> G = NeuronGroup(10, 'v : volt')
    >>> v_var = G.variables['v']
    >>> print(device.array_cache[v_var])  # CPPStandaloneDevice.init_with_zeros stored initial zero values
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    >>> G.v = -70*mV
    >>> print(device.array_cache[v_var])  # CPPStandaloneDevice.fill_with_array updated the values
    [-0.07 -0.07 -0.07 -0.07 -0.07 -0.07 -0.07 -0.07 -0.07 -0.07]
    >>> G.v = '-70*mV + i*2*mV'
    >>> print(device.array_cache[v_var])  # Array cache for v has been invalidated
    None
