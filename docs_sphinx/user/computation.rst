Computational methods and efficiency
====================================

.. contents::
    :local:
    :depth: 2

Brian has several different methods for running the computations in a
simulation. The default mode is :ref:`runtime`, which runs the simulation loop
in Python but compiles and executes the modules doing the actual simulation
work (numerical integration, synaptic propagation, etc.) in a defined target
language. Brian will select the best available target language automatically.
On Windows, to ensure that you get the advantages of compiled code, read
the instructions on installing a suitable compiler in
:ref:`installation_cpp`.
Runtime mode has the advantage that you can combine the computations
performed by Brian with arbitrary Python code specified as `NetworkOperation`.

The fact that the simulation is run in Python means that there is a (potentially
big) overhead for each simulated time step. An alternative is to run Brian in with
:ref:`cpp_standalone` -- this is in general faster (for certain types of simulations
*much* faster) but cannot be used for all kinds of simulations. To enable this
mode, add the following line after your Brian import, but before your simulation
code::

    set_device('cpp_standalone')

For detailed control over the compilation process (both for runtime and standalone
code generation), you can change the :ref:`compiler_settings` that are used.

.. admonition:: The following topics are not essential for beginners.

    |

.. _runtime:

Runtime code generation
-----------------------
Code generation means that Brian takes the Python code and strings
in your model and generates code in one of several possible different
languages which is then executed. The target language for this code
generation process is set in the `codegen.target` preference. By default, this
preference is set to ``'auto'``, meaning that it will choose the compiled language
target if possible and fall back to Python otherwise (also raising a warning).
The compiled language target is ``'cython'`` which needs the `Cython`_ package in
addition to a working C++ compiler. If you want to
chose a code generation target explicitly (e.g. because you want to get rid of the
warning that only the Python fallback is available), set the preference to ``'numpy'``
or ``'cython'`` at the beginning of your script::

    from brian2 import *
    prefs.codegen.target = 'numpy'  # use the Python fallback

See :doc:`../advanced/preferences` for different ways of setting preferences.

 .. _Cython: http://cython.org/

Caching
~~~~~~~
When you run code with ``cython`` for the first time, it will take
some time to compile the code. For short simulations, this can make these
targets to appear slow compared to the ``numpy`` target where such compilation
is not necessary. However, the compiled code is stored on disk and will be
re-used for later runs, making these simulations start faster. If you run many
simulations with different code (e.g. Brian's
:doc:`test suite <../developer/guidelines/testing>`), this code can take quite
a bit of space on the disk. During the import of the ``brian2`` package, we
check whether the size of the disk cache exceeds the value set by the
`codegen.max_cache_dir_size` preference (by default, 1GB) and display a message
if this is the case. You can clear the disk cache manually, or use the
`~brian2.__init__.clear_cache` function, e.g. ``clear_cache('cython')``.

.. note::

    If you run simulations on parallel on a machine using the Network File System, see
    :ref:`this known issue <parallel_cython>`.

.. _cpp_standalone:

Standalone code generation
--------------------------
Brian supports generating standalone code for multiple devices. In this mode, running a Brian script generates
source code in a project tree for the target device/language. This code can then be compiled and run on the device,
and modified if needed. At the moment, the only "device" supported is standalone C++ code.
In some cases, the speed gains can be impressive, in particular for smaller networks with complicated spike
propagation rules (such as STDP).

To use the C++ standalone mode, you only have to make very small changes to your script. The exact change depends on
whether your script has only a single `run` (or `Network.run`) call, or several of them:

Single run call
~~~~~~~~~~~~~~~
At the beginning of the script, i.e. after the import statements, add::

    set_device('cpp_standalone')

The `Device.build` function will be automatically called with default arguments right after the `run`
call. If you need non-standard arguments then you can specify them as part of the `set_device` call::

    set_device('cpp_standalone', directory='my_directory', debug=True)

Multiple run calls
~~~~~~~~~~~~~~~~~~
At the beginning of the script, i.e. after the import statements, add::

    set_device('cpp_standalone', build_on_run=False)

After the last `run` call, call `CPPStandaloneDevice.build` explicitly::

    device.build()

The `~Device.build` function has several arguments to specify the output directory, whether or not to
compile and run the project after creating it and whether or not to compile it with debugging support or not.

.. _standalone_multiple_full_runs:

Multiple full simulation runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run multiple full, independent, simulations (i.e. not just multiple `run` calls as discussed above), you can 
use the device's `~brian2.devices.cpp_standalone.device.CPPStandaloneDevice.run` function after an initial build. This will use the previously
generated and compiled code, and will therefore run immediately. Note that you cannot change the model or its
parameters in the usual way between the `~brian2.devices.cpp_standalone.device.CPPStandaloneDevice.build` and
`~brian2.devices.cpp_standalone.device.CPPStandaloneDevice.run` calls.
If you want to change some of its parameters, you will have to use the ``run_args`` argument as described below.

Running multiple simulations with same parameters
+++++++++++++++++++++++++++++++++++++++++++++++++

By default, a device's `~brian2.devices.cpp_standalone.device.CPPStandaloneDevice.run` will run the simulation again,
using the same model parameters and initializations. This can be useful, when the model is itself stochastic
(e.g. using the ``xi`` noise term  in the equations, using a stochastic group such as `PoissonGroup` or
`PoissonInput`, etc.), when it uses random synaptic connections, or when it uses random variable initialization::

    set_device('cpp_standalone')
    group = NeuronGroup(1, 'dv/dt = -v / (10*ms) : 1')  # a simple IF neuron without threshold
    group.v = 'rand()'  # v is randomly initialized between 0 and 1
    mon = StateMonitor(group, 'v', record=0)
    run(100*ms)  # calls device.build and device.run
    results = [mon.v[0]]
    # Do 9 more runs without recompiling, each time initializing v to a new value
    for _ in range(9):
        device.run()
        results.append(mon.v[0])

For more consistent code, you might consider to disable the automatic ``device.build``/``device.run`` call, so
that the initial run of the simulation is not different to subsequent runs::

    set_device('cpp_standalone', build_on_run=False)
    # ... Set up model as before
    run(100*ms)  # will not call device.build/device.run
    device.build(run=False)  # Compile the code
    results = []
    # Do 10 runs without recompiling, each time initializing v to a new value
    for _ in range(10):
        device.run()
        results.append(mon.v[0])


Running multiple simulations with different parameters
++++++++++++++++++++++++++++++++++++++++++++++++++++++

When launching new simulation runs as described above, you can also change parameters of the model. Note that this
only concerns parameters that are included in equations, you cannot change externally defined constants. You can
easily work around this limitation, however, by declaring such constants in the equations, using the ``(shared, constant)``
flags. Here's a similar example to the one shown before, now exploring the effect of the time constant ``tau``, while
assuring via a `seed` call that the random initializations are identical across runs::

    set_device('cpp_standalone', build_on_run=False)
    seed(111)  # same random numbers for each run
    group = NeuronGroup(10, '''dv/dt = -v / tau : 1
                               tau : second (shared, constant)''')  # 10 simple IF neuron without threshold
    group.v = 'rand()'
    mon = StateMonitor(group, 'v', record=0)
    run(100*ms)
    device.build(run=False)  # Compile the code
    results = []
    # Do 10 runs without recompiling, each time setting group.tau to a new value
    for tau_value in (np.arange(10)+1)*5*ms:
        device.run(run_args={group.tau: tau_value})
        results.append(mon.v[:])

You can use the same mechanism to provide an array of initial values for a group. E.g., to systematically try out
different initializations of ``v``, you could use::

    set_device('cpp_standalone', build_on_run=False)
    group = NeuronGroup(10, 'dv/dt = -v / (10*ms) : 1')  # ten simple IF neurons without threshold
    mon = StateMonitor(group, 'v', record=True)
    run(100*ms)  # will not call device.build/device.run
    device.build(run=False)  # Compile the code
    results = []
    # Do 10 runs without recompiling, each time initializing v differently
    for idx in range(10):
        device.run(run_args={group.v: np.arange(10)*0.01 + 0.1*idx})
        results.append(mon.v[0])

You can also overwrite the values in a `TimedArray` using this mechanism, by using the `TimedArray` as a key in the
``run_args`` dictionary::

    set_device('cpp_standalone', build_on_run=False)
    stim = TimedArray(np.zeros(10), dt=10*ms)
    group = NeuronGroup(10, 'dv/dt = (stim(t) - v)/ (10*ms) : 1')  # time-dependent stimulus
    mon = StateMonitor(group, 'v', record=True)
    run(100 * ms)
    device.build(run=False)
    results = []
    # Do 10 runs with a 10ms at a random time
    for idx in range(10):
        values = np.zeros(10)
        values[np.random.randint(0, 10)] = 1
        device.run(run_args={stim: values})
        results.append(mon.v[0])

By default, the initialization provided via ``run_args`` overwrites any initializations done in the usual way.
This might not exactly do what you want if you use string-based variable initializations that refer to each other.
For example, if your equations contain two synaptic time constants ``tau_exc`` and ``tau_inh``, and you always
want the latter to be twice the value of the former, you can write::

    group.tau_exc = 5*ms
    group.tau_inh = 'tau_exc * 2'

If you now use the ``run_args`` argument to set ``tau_exc`` to a different value, this will not be taken into account
for setting ``tau_inh``, since the value change for ``tau_exc`` happens *after* the initialization of ``tau_inh``.
Of course you can simply set the value for ``tau_inh`` manually using ``run_args`` as well, but a more general solution
is to move the point where the ``run_args`` are applied. You can do this by calling the device's
`~brian2.devices.cpp_standalone.device.CPPStandaloneDevice.apply_run_args` function::

    group.tau_exc = 5*ms
    device.apply_run_args()
    group.tau_inh = 'tau_exc * 2'

With this change, setting ``tau_exc`` via ``run_args`` will affect the value of ``tau_inh``.

Running multiple simulations in parallel
++++++++++++++++++++++++++++++++++++++++

The techniques mentioned above cannot be directly used to run simulations in parallel (e.g. with Python's
`multiprocessing` module), since all of them will try to write the results to the same place. You can
circumvent this problem by specifying the ``results_directory`` argument, and setting it to a different value
for each run. Note that using the standalone device with `multiprocessing` can be a bit tricky, since the
currently selected device is stored globally in the ``device`` module. Use the approach presented below to
make sure the device is selected correctly. Here's a variant of the previously shown example running a
simulation with random initialization repeatedly, this time running everything in parallel using Python's
`multiprocessing` module::

    class SimWrapper:
        def __init__(self):
            # Runs once to set up the simulation
            group = NeuronGroup(1, 'dv/dt = -v / (10*ms) : 1', name='group')
            group.v = 'rand()'  # v is randomly initialized between 0 and 1
            mon = StateMonitor(group, 'v', record=0, name='monitor')
            # Store everything in a network
            self.network = Network([group, mon])
            self.network.run(100*ms)
            device.build(run=False)
            self.device = get_device()  # store device object

        def do_run(self, result_dir):
            # Runs in every process
            # Workaround to set the device globally in this context
            from brian2.devices import device_module
            device_module.active_device = self.device
            self.device.run(results_directory=result_dir)
            # Return the results
            return self.network['monitor'].v[0]
    
    if __name__ == '__main__':  # Important for running on Windows and OS X
        set_device('cpp_standalone', build_on_run=False)
        sim = SimWrapper()
        import multiprocessing
        with multiprocessing.Pool() as p:
            # Run 10 simulations in parallel
            results = p.map(sim.do_run, [f'result_{idx}' for idx in range(10)])

You can also use parallel runs with the ``run_args`` argument. For example, to do 10 simulations
with different (deterministic) initial values for ``v``::

    class SimWrapper:
        # ... model definition without random initialization

        def do_run(self, v_init):
            # Set result directory based on variable
            result_dir = f'result_{v_init}'
            self.device.run(run_args={self.network['group'].v: v_init},
                            results_directory=result_dir)
            # Return the results
            return self.network['monitor'].v[0]
    
    if __name__ == '__main__':  # Important for running on Windows and OS X
        set_device('cpp_standalone', build_on_run=False)
        sim = SimWrapper()
        import multiprocessing
        with multiprocessing.Pool() as p:
            # Run 10 simulations in parallel
            results = p.map(sim.do_run, np.linspace(0, 1, 10))


Limitations
~~~~~~~~~~~
Not all features of Brian will work with C++ standalone, in particular Python based network operations and
some array based syntax such as ``S.w[0, :] = ...`` will not work. If possible, rewrite these using string
based syntax and they should work. Also note that since the Python code actually runs as normal, code that does
something like this may not behave as you would like::

    results = []
    for val in vals:
        # set up a network
        run()
        results.append(result)

The current C++ standalone code generation only works for a fixed number of `~Network.run` statements, not with loops.
If you need to do loops or other features not supported automatically, you can do so by inspecting the generated
C++ source code and modifying it, or by inserting code directly into the main loop as described below.

.. _standalone_variables:

Variables
~~~~~~~~~
In standalone mode, code will only be executed when the simulation is run (after the `run` call by default, or after a call
to `~brian2.devices.cpp_standalone.device..build`, if `set_device` has been called with ``build_on_run`` set to ``False``).
This means that it is not possible to access state variables and synaptic connection indices in the Python script doing the
set up of the model. For example, the following code would work fine in runtime mode, but raise a ``NotImplementedError``
in standalone mode::

    neuron = NeuronGroup(10, 'v : volt')
    neuron.v = '-70*mV + rand()*10*mV'
    print(np.mean(neuron.v))

Sometimes, access is needed to make one variable depend on another variable for initialization. In such cases, it is often
possible to circumvent the issue by using initialization with string expressions for both variables. For example, to set
the initial membrane potential relative to a random leak reversal potential, the following code would work in runtime mode
but fail in standalone mode::

    neuron = NeuronGroup(10, 'dv/dt = -g_L*(v - E_L)/tau : volt')
    neuron.E_L = '-70*mV + rand()*10*mV'  # E_L between -70mV and -60mV
    neuron.v = neuron.E_L  # initial membrane potential equal to E_L

Instead, you can initialize the variable `v` with a string expression, which means that standalone will execute it
during the run when the value of `E_L` is available::

    neuron = NeuronGroup(10, 'dv/dt = -g_L*(v - E_L)/tau : volt')
    neuron.E_L = '-70*mV + rand()*10*mV'  # E_L between -70mV and -60mV
    neuron.v = 'E_L'  # works both in runtime and standalone mode

The same applies to synaptic indices. For example, if we want to set weights differently depending on the target index
of a synapse, the following would work in runtime mode but fail in standalone mode, since the synaptic indices have not
been determined yet::

    neurons = NeuronGroup(10, '')
    synapses = Synapses(neurons, neurons, 'w : 1')
    synapses.connect(p=0.25)
    # Set weights to low values when targetting first five neurons and to high values otherwise
    synapses.w[:, :5] = 0.1
    synapses.w[:, 5:] = 0.9

Again, this initialization can be replaced by string expressions, so that standalone mode can evaluate them in the
generated code after synapse creation::

    neurons = NeuronGroup(10, '')
    synapses = Synapses(neurons, neurons, 'w : 1')
    synapses.connect(p=0.25)
    # Set weights to low values when targetting first five neurons and to high values otherwise
    synapses.w['j < 5'] = 0.1
    synapses.w['j >= 5'] = 0.9

Note that this limitation only applies if the variables or synapses have been initialized in ways that require the
execution of code. If instead they are initialized with concrete values, they can be accessed in Python code even
in standalone mode::

    neurons = NeuronGroup(10, 'v : volt')
    neurons.v = -70*mV 
    print(np.mean(neurons.v))  # works in standalone
    synapses = Synapses(neurons, neurons, 'w : 1')
    synapses.connect(i=[0, 2, 4, 6, 8], j=[1, 3, 5, 7, 9])
    # works as well, since synaptic indices are known
    synapses.w[:, :5] = 0.1
    synapses.w[:, 5:] = 0.9



In any case, state variables, synaptic indices, and monitored variables can be accessed using standard syntax *after* a
run (with a few exceptions, e.g. string expressions for indexing).

.. _openmp:

Multi-threading with OpenMP
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
    OpenMP code has not yet been well tested and so may be inaccurate.

When using the C++ standalone mode, you have the opportunity to turn on multi-threading, if your C++ compiler is compatible with
OpenMP. By default, this option is turned off and only one thread is used. However, by changing the preferences of the codegen.cpp_standalone
object, you can turn it on. To do so, just add the following line in your python script::

    prefs.devices.cpp_standalone.openmp_threads = XX

XX should be a positive value representing the number of threads that will be
used during the simulation. Note that the speedup will strongly depend on the
network, so there is no guarantee that the speedup will be linear as a function
of the number of threads. However, this is working fine for networks with not
too small timestep (dt > 0.1ms), and results do not depend on the number of
threads used in the simulation.

Custom code injection
~~~~~~~~~~~~~~~~~~~~~
It is possible to insert custom code directly into the generated code of a
standalone simulation using a Device's `~.Device.insert_code` method::

    device.insert_code(slot, code)

``slot`` can be one of ``main``, ``before_start``, ``after_start``,
``before_network_run``, ``after_network_run``, ``before_end`` and ``after_end``,
which determines where the code is inserted. ``code`` is the code in the
Device's language. Here is an example for the C++ Standalone Device::

    device.insert_code('main', '''
    cout << "Testing direct insertion of code." << endl;
    ''')

For the C++ Standalone Device, all code is inserted into the ``main.cpp`` file,
here into the ``main`` slot, referring to the main simulation function.
This is a simplified version of this function in ``main.cpp``::

    int main(int argc, char **argv)
    {
        // before_start
        brian_start();
        // after_start

        {{main_lines}}

        // before_end
        brian_end();
        // after_end

        return 0;
    }

``{{main_lines}}`` is replaced in the generated code with the actual simulation.
Code inserted into the ``main`` slot will be placed within the
``{{main_lines}}``. ``brian_start`` allocates and initializes all arrays needed
during the simulation and ``brian_end`` writes the results to disc and
deallocates memory. Within the ``{{main_lines}}``, all ``Network`` objects
defined in Python are created and run. Code inserted in the
``before/after_network_run`` slot will be inserted around the ``Network.run``
call, which starts the time loop. Note that if your Python script has multiple
``Network`` objects or multiple ``run`` calls, code in the
``before/after_network_run`` slot will be inserted around each ``Network.run``
call in the generated code.

The code injection mechanism has been used for benchmarking experiments, see
e.g. `here for Brian2CUDA benchmarks <https://github.com/brian-team/brian2cuda/blob/835c978ad758bc0621e34344c1fb7b811ef8a118/brian2cuda/tests/features/cuda_configuration.py#L148-L156>`_ or `here for Brian2GeNN benchmarks <https://github.com/brian-team/brian2genn_benchmarks/blob/6d1a6d9d97c05653cec2e413c9fd312cfe13e15c/benchmark_utils.py#L78-L136>`_.


.. _standalone_custom_build:

Customizing the build process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In standalone mode, a standard "make file" is used to orchestrate the
compilation and linking. To provide additional arguments to the ``make`` command
(respectively ``nmake`` on Windows), you can use the
`devices.cpp_standalone.extra_make_args_unix` or
`devices.cpp_standalone.extra_make_args_windows` preference. On Linux,
this preference is by default set to ``['-j']`` to enable parallel compilation.
Note that you can also use these arguments to overwrite variables in the make
file, e.g. to use `clang <https://clang.llvm.org/>`_ instead of the default
`gcc <https://gcc.gnu.org/>`_ compiler::

    prefs.devices.cpp_standalone.extra_make_args_unix += ['CC=clang++']


.. _compiler_settings:

Cleaning up after a run
~~~~~~~~~~~~~~~~~~~~~~~
Standalone simulations store all results of a simulation (final state variable
values and values stored in monitors) to disk. These results can take up quite
significant amount of space, and you might therefore want to delete these
results when you do not need them anymore. You can do this by using the device's
`~.Device.delete` method::

    device.delete()

Be aware that deleting the data will make all access to state variables fail,
including the access to values in monitors. You should therefore only delete the
data after doing all analysis/plotting that you are interested in.

By default, this function will delete both the generated code and the data, i.e.
the full project directory. If you want to keep the code (which typically takes
up little space compared to the results), exclude it from the deletion::

    device.delete(code=False)

If you added any additional files to the project directory manually, these will
not be deleted by default. To delete the full directory regardless of its
content, use the ``force`` option::

    device.delete(force=True)

.. note::
    When you initialize state variables with concrete values (and not with
    a string expression), they will be stored to disk from your Python script
    and loaded from disk at the beginning of the standalone run. Since these
    values are necessary for the compiled binary file to run, they are
    considered "code" from the point of view of the `~.Device.delete` function.

Compiler settings
-----------------

If using C++ code generation (either via cython or standalone), the
compiler settings can make a big difference for the speed of the simulation.
By default, Brian uses a set of compiler settings that switches on various
optimizations and compiles for running on the same architecture where the
code is compiled. This allows the compiler to make use of as many advanced
instructions as possible, but reduces portability of the generated executable
(which is not usually an issue).

If there are any issues with these compiler settings, for example because
you are using an older version of the C++ compiler or because you want to
run the generated code on a different architecture, you can change the
settings by manually specifying the `codegen.cpp.extra_compile_args`
preference (or by using `codegen.cpp.extra_compile_args_gcc` or
`codegen.cpp.extra_compile_args_msvc` if you want to specify the settings
for either compiler only).
