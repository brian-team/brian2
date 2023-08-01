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
    >>> set_device('runtime')  # Reset device to avoid problems in other doctests

Command line arguments
----------------------

The mechanisms described in :ref:`standalone_multiple_full_runs` are implemented via command-line arguments
to the ``main`` binary. A call such as::

    device.run(results_directory='results_1', run_args={group.tau : 10*ms})

will be executed by calling the compiled binary as follows:

.. code-block:: verbatim

    ./main --results_dir /full/path/to/results_1 neurongroup.tau=0.01

where ``neurongroup`` is ``group.name`` (either a default name, e.g. ``neurongroup``, or the name
set during construction with the ``name`` argument). The generated code applies the initialization
for ``tau`` after the usual variable initializations, before the call of ``network.run``
(assuming that application has not been moved to a different place by using
`~brian2.devices.cpp_standalone.device.CPPStandaloneDevice.apply_run_args`).

For initializations with a vector of values, the values are written to disk (in the same, simple
binary format that is used elsewhere, e.g. for the results). The command line argument then
specifies the file name instead of a value:

.. code-block:: verbatim

    ./main neurongroup.tau=static_arrays/init_neurongroup_v_aca4cd6a3f7e526a61bb5a07468b377e.dat

The hex string in the filename is an automatically generated MD5 hash of the array content. This
makes it possible to assure that each array is only written to disk once, even for repeated and
parallel executions with the same values (file locking is used to make sure only one process
writes to each file).
