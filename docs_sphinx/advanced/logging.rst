Logging
=======

Brian uses a logging system to display warnings and general information messages
to the user, as well as writing them to a file with more detailed information,
useful for debugging. Each log message has one of the following "log levels":

``ERROR``
    Only used when an exception is raised, i.e. an error occurs and the current
    operation is interrupted. *Example:* You use a variable name in an equation
    that Brian does not recognize.

``WARNING``
    Brian thinks that something is most likely a bug, but it cannot be sure.
    *Example:* You use a `Synapses` object without any synapses in your
    simulation.

``INFO``
    Brian wants to make the user aware of some automatic choice that it did for
    the user. *Example:* You did not specify an integration ``method`` for a
    `NeuronGroup` and therefore Brian chose an appropriate method for you.

``DEBUG``
    Additional information that might be useful when a simulation is not working
    as expected. *Example:* The integration timestep used during the simulation.

``DIAGNOSTIC``
    Additional information useful when tracking down bugs in Brian itself.
    *Example:* The generated code for a `CodeObject`.

By default, all messages with level ``DEBUG`` or above are written to the log file
and all messages of level ``INFO`` and above are displayed on the console. To change
what messages are displayed, see below.

.. note:: By default, the log file is deleted after a successful simulation run,
   i.e. when the simulation exited without an error. To keep the log around,
   set the `logging.delete_log_on_exit` preference to ``False``.

Logging and multiprocessing
---------------------------
Brian's logging system is not designed for multiple parallel Brian processes
started via Python's `multiprocessing` module (see the
:ref:`multiprocessing examples <multiprocessing>`). Log messages that get printed
from different processes to the console are not printed in a well-defined order and
do not contain any indication about which processes they are coming from. You might
therefore consider using e.g. `BrianLogger.log_level_error()` to only show error
messages before starting the processes and avoid cluttering your console with
warning and info messages.

To avoid issues when multiple processes try to log to the same log file, file
logging is automatically switched off for all processes except for the initial
process. If you need a file log for sub-processes, you can call
`BrianLogger.initialize()` in each sub-process. This way, each process will log
to its own file.

Showing/hiding log messages
---------------------------
If you want to change what messages are displayed on the console, you can call a
method of the method of `BrianLogger`::

    BrianLogger.log_level_debug() # now also display debug messages

It is also possible to suppress messages for certain sub-hierarchies by using
`BrianLogger.suppress_hierarchy`::

    # Suppress code generation messages on the console
    BrianLogger.suppress_hierarchy('brian2.codegen')
    # Suppress preference messages even in the log file
    BrianLogger.suppress_hierarchy('brian2.core.preferences',
                                   filter_log_file=True)

Similarly, messages ending in a certain name can be suppressed with
`BrianLogger.suppress_name`::

    # Suppress resolution conflict warnings
    BrianLogger.suppress_name('resolution_conflict')

These functions should be used with care, as they suppresses messages
independent of the level, i.e. even warning and error messages.

Preferences
-----------
You can also change details of the logging system via Brian's :doc:`preferences`
system. With this mechanism, you can switch the logging to a file off completely
(by setting `logging.file_log` to ``False``) or have it log less messages (by
setting `logging.file_log_level` to a level higher than ``DEBUG``). To debug
details of the code generation system, you can also set `logging.file_log_level`
to ``DIAGNOSTIC``. Note that this will make the log file grow quickly in size. To
prevent it from filling up the disk, it will only be allowed to grow up to a certain
size. You can configure the maximum file size with the `logging.file_log_max_size`
preference.

For a list of all preferences related to logging, see the documentation of the
`brian2.utils.logger` module.

.. warning:: Most of the logging preferences are only taken into account during
   the initialization of the logging system which takes place as soon as `brian2`
   is imported. Therefore, if you use e.g. `prefs.logging.file_log = False` in
   your script, this will not have the intended effect! To make sure these
   preferences are taken into account, call `BrianLogger.initialize` after
   setting the preferences. Alternatively, you can set the preferences in a file
   (see :doc:`preferences`).
