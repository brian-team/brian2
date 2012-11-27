Brian compiler
==============
There are two ways in which code generation can be used.
The main way is to use Brian as a host that communicates with a device. Essentially, it executes code on a device,
but the main loop is controlled by Brian.
Another possible use is to run an entire simulation on a device. That is, Brian is used only to produce code that
is then executed on the device, including the main loop. One application is real-time simulations, e.g. embedded
on robots.

In the host-device configuration, there are different pieces of code that are executed. For example, at construction
time, synaptic connections can be built with code. Each code string is compiled and directly executed on the device.
In the compiler configuration, the code string is transformed into device code, but this is not executed. It is only
stored, pushed to a stack of statements. Then the run() statement gathers code for all objects in the network and produces
a complete code.

Compiler configuration
----------------------
The compiler configuration could be set with set_global_preference() at the beginning of the script.

Construction
^^^^^^^^^^^^
The main difficulty I see with the compiler configuration mode is the construction of the network.
There are two ways to handle this:

* Construction is done offline by Brian. Then when the run() method is called, the state of the network
  (synaptic weights, neuron variables etc) is saved to a file. The code produced by run() includes an
  initial loading function. In this configuration, we may want to use a specific method, e.g. compile(),
  and the global preference is not useful.
* Construction is done on device. This means that every assignment should only produce a code that is
  accumulated, rather than executing it. This restricts the type of statements that can be done (for example
  loops over all synapses), but it may still be practical thanks to assignments with code strings. An error
  could be issued when it is not possible to produce code for an assignment.
  Then the run() statement would also only produce a code. To obtain the code, we may have a special
  instruction at the end of the script (e.g. compile() or save_code()). In this way, it would be possible
  to have several run() functions in the same script.

External interactions
^^^^^^^^^^^^^^^^^^^^^
For embedded simulations on robots, the script must communicate with the external world. Because this
is device specific, it should be provided externally and linked to monitor objects in Brian.
