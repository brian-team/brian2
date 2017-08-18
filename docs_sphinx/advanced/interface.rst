Interfacing with external code
==============================

Some neural simulations benefit from a direct connections to external libraries,
e.g. to support real-time input from a sensor (but note that Brian currently
does not offer facilities to assure real-time processing) or to perform
complex calculations during a simulation run.

If the external library is written in Python (or is a library with Python
bindings), then the connection can be made either using the mechanism for
:ref:`user_functions`, or using a :ref:`network operation <network_operation>`.

In case of C/C++ libraries, only the :ref:`user_functions` mechanism can be
used. On the other hand, such simulations can use the same user-provided C++
code to run both with the runtime ``weave`` target and with the
:ref:`cpp_standalone` mode. In addition to that code, one generally needs to
include additional header files and use compiler/linker options to interface
with the external code. For this, several preferences can be used that will be
taken into account for ``weave``, ``cython`` and the ``cpp_standalone`` device.
These preferences are mostly equivalent to the respective keyword arguments for
Python's `distutils.core.Extension` class, see the documentation of the
`~brian2.codegen.cpp_prefs` module for more details.
