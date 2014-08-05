Devices
=======

Brian 2 supports generating standalone code for multiple devices. In this mode, running a Brian script generates
source code in a project tree for the target device/language. This code can then be compiled and run on the device,
and modified if needed. At the moment, the only 'device' supported is standalone C++ code.
In some cases, the speed gains can be impressive, in particular for smaller networks with complicated spike
propagation rules (such as STDP).

C++ standalone
--------------

To use the C++ standalone mode, make the following changes to your script:

1. At the beginning of the script, i.e. after the import statements, add::

    set_device('cpp_standalone')

2. After ``run(duration)`` in your script, add::

    device.build(project_dir='output', compile_project=True, run_project=True, debug=False)

The `~CPPStandaloneDevice.build` function has several arguments to specify the output directory, whether or not to compile and run
the project after creating it (using ``gcc``) and whether or not to compile it with debugging support or not.

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
C++ source code and modifying it, or by inserting code directly into the main loop as follows::

    device.insert_device_code('main.cpp', '''
    cout << "Testing direct insertion of code." << endl;
    ''')

After a simulation has been run (using the ``run_project`` keyword in the `Device.build` call), state variables and
monitored variables can be accessed using standard syntax, with a few exceptions (e.g. string expressions for indexing).
