.. currentmodule:: brian2

Logging
=======

For a description of logging from the users point of view, see :doc:`../../advanced/logging`.

Logging in Brian is based on the :mod:`logging` module in Python's standard
library.

Every brian module that needs logging should start with the following line,
using the `get_logger` function to get an instance of `BrianLogger`::

    logger = get_logger(__name__)

In the code, logging can then be done via::

    logger.diagnostic('A diagnostic message')
    logger.debug('A debug message')
    logger.info('An info message')
    logger.warn('A warning message')
    logger.error('An error message')

If a module logs similar messages in different places or if it might be useful
to be able to suppress a subset of messages in a module, add an additional
specifier to the logging command, specifying the class or function name, or
a method name including the class name (do not include the module name, it will
be automatically added as a prefix)::

    logger.debug('A debug message', 'CodeString')
    logger.debug('A debug message', 'NeuronGroup.update')
    logger.debug('A debug message', 'reinit')

If you want to log a message only once, e.g. in a function that is called
repeatedly, set the optional ``once`` keyword to ``True``::

    logger.debug('Will only be shown once', once=True)
    logger.debug('Will only be shown once', once=True)

The output of debugging looks like this in the log file::

    2012-10-02 14:41:41,484 DEBUG    brian2.equations.equations.CodeString: A debug message

and like this on the console (if the log level is set to "debug")::

    DEBUG    A debug message [brian2.equations.equations.CodeString]

.. _log_level_recommendations:

Log level recommendations
-------------------------
diagnostic
    Low-level messages that are not of any interest to the normal user but
    useful for debugging Brian itself. A typical example is the source
    code generated by the code generation module.
debug
    Messages that are possibly helpful for debugging the user's code. For example,
    this shows which objects were included in the network, which clocks the
    network uses and when simulations start and stop.

info
    Messages which are not strictly necessary, but are potentially helpful for
    the user. In particular, this will show messages about the chosen state
    updater and other information that might help the user to achieve better
    performance and/or accuracy in the simulations (e.g. using ``(event-driven)``
    in synaptic equations, avoiding incompatible ``dt`` values between
    `TimedArray` and the `NeuronGroup` using it, ...)
warn
    Messages that alert the user to a potential mistake in the code, e.g. two
    possible resolutions for an identifier in an equation. In such cases, the
    warning message should include clear information how to change the code
    to make the situation unambigous and therefore make the warning message
    disappear. It can also be used to make the user aware that he/she is using
    an experimental feature, an unsupported compiler or similar. In this case,
    normally the ``once=True`` option should be used to raise this warning only
    once. As a rule of thumb, "common" scripts like the examples provided in
    the examples folder should normally not lead to any warnings.
error
    This log level is not used currently in Brian, an exception should be
    raised instead. It might be useful in "meta-code", running scripts and
    catching any errors that occur.

The default log level shown to the user is ``info``. As a general rule, all
messages that the user sees in the default configuration (i.e., ``info`` and
``warn`` level) should be avoidable by simple changes in the user code, e.g.
the renaming of variables, explicitly specifying a state updater instead of
relying on the automatic system, adding ``(clock-driven)``/``(event-driven)``
to synaptic equations, etc.

Testing log messages
--------------------
It is possible to test whether code emits an expected log message using the
`~brian2.utils.logger.catch_logs` context manager. This is normally not
necessary for debug and info messages, but should be part of the unit tests
for warning messages (`~brian2.utils.logger.catch_logs` by default only catches
warning and error messages)::

    with catch_logs() as logs:
        # code that is expected to trigger a warning
        # ...
        assert len(logs) == 1
        # logs contains tuples of (log level, name, message)
        assert logs[0][0] == 'WARNING' and logs[0][1].endswith('warning_type')

Logging in extension packages
-----------------------------
Extension packages such as `brian2cuda <https://brian2cuda.readthedocs.io>`_ can use Brian's logging infrastructure by
using the `~brian2.utils.logger.get_logger` function to get a logger instance. They should use their own module name,
e.g. a name starting with ``brian2cuda.`` so that it is clear whether a log message comes from Brian or from an
extension package. This is also used by the `~.catch_logs` context manager (see above) to only consider log messages
from the ``brian2`` package.