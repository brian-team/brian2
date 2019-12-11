Testing
=======

Brian uses the `pytest package <https://docs.pytest.org/>`__
for its testing framework.

Running the test suite
----------------------
The pytest tool automatically finds tests in the code. However, to deal with the
different code generation targets, and correctly set up tests for standalone mode, it is
recommended to use Brian's builtin test function that calls pytest appropriately::

	>>> import brian2
	>>> brian2.test()  # doctest: +SKIP

By default, this runs the test suite for all available (runtime) code generation
targets. If you only want to test a specific target, provide it as an argument::

    >>> brian2.test('numpy')  # doctest: +SKIP

If you want to test several targets, use a list of targets::

    >>> brian2.test(['weave', 'cython'])  # doctest: +SKIP


In addition to the tests specific to a code generation target, the test suite
will also run a set of independent tests (e.g. parsing of equations, unit
system, utility functions, etc.). To exclude these tests, set the
``test_codegen_independent`` argument to ``False``. Not all available tests are
run by default, tests that take a long time are excluded. To include these, set
``long_tests`` to ``True``.

To run the C++ standalone tests, you have to set the ``test_standalone``
argument to the name of a standalone device. If you provide an empty argument
for the runtime code generation targets, you will only run the standalone
tests::

    >>> brian2.test([], test_standalone='cpp_standalone')  # doctest: +SKIP


Writing tests
-------------
Generally speaking, we aim for a 100% code coverage by the test suite. Less
coverage means that some code paths are never executed so there's no way of
knowing whether a code change broke something in that path.

Unit tests
~~~~~~~~~~
The most basic tests are unit tests, tests that test one kind of functionality or
feature. To write a new unit test, add a function called ``test_...`` to one of
the ``test_...`` files in the ``brian2.tests`` package. Test files should
roughly correspond to packages, test functions should roughly correspond to
tests for one function/method/feature. In the test functions, use assertions
that will raise an ``AssertionError`` when they are violated, e.g.::

    G = NeuronGroup(42, model='dv/dt = -v / (10*ms) : 1')
    assert len(G) == 42

When comparing arrays, use the `array_equal` function from
`numpy.testing.utils` which takes care of comparing types, shapes and content
and gives a nicer error message in case the assertion fails. Never make tests
depend on external factors like random numbers -- tests should always give the
same result when run on the same codebase. You should not only test the
expected outcome for the correct use of functions and classes but also that
errors are raised when expected. For that you can use pytest's ``raises``
function with which you can define a block of code that should raise an exception of
a certain type::

    with pytest.raises(DimensionMismatchError):
        3*volt + 5*second

You can also check whether expected warnings are raised, see the documentation of the
:doc:`logging mechanism <logging>` for details

For simple functions, doctests (see below) are a great alternative to writing
classical unit tests.

By default, all tests are executed for all selected runtime code generation
targets (see `Running the test suite`_ above). This is not useful for all tests,
some basic tests that for example test equation syntax or the use of physical
units do not depend on code generation and need therefore not to be repeated. To
execute such tests only once, they can be annotated with a
``codegen_independent`` marker, using the `~pytest.mark`
decorator::

    import pytest
    from brian2 import NeuronGroup

    @pytest.mark.codegen_independent
    def test_simple():
        # Test that the length of a NeuronGroup is correct
        group = NeuronGroup(5, '')
        assert len(group) == 5

Tests that are not "codegen-independent" are by default only executed for the
runtimes device, i.e. not for the ``cpp_standalone`` device, for example.
However, many of those tests follow a common pattern that is compatible with
standalone devices as well: they set up a network, run it, and check the state
of the network afterwards. Such tests can be marked as
``standalone_compatible``, using the `~pytest.mark` decorator in
the same way as for ``codegen_independent`` tests.::

    import pytest
    from numpy.testing.utils import assert_equal
    from brian2 import *

    @pytest.mark.standalone_compatible
    def test_simple_run():
        # Check that parameter values of a neuron don't change after a run
        group = NeuronGroup(5, 'v : volt')
        group.v = 'i*mV'
        run(1*ms)
        assert_equal(group.v[:], np.arange(5)*mV)

Tests that have more than a single run function but are otherwise compatible
with standalone mode (e.g. they don't need access to the number of synapses or
results of the simulation before the end of the simulation), can be marked as
``standalone_compatible`` and ``multiple_runs``. They then have to use an
explicit ``device.build(...)`` call of the form shown below::

    import pytest
    from numpy.testing.utils import assert_equal
    from brian2 import *

    @pytest.mark.standalone_compatible
    @pytest.mark.multiple_runs
    def test_multiple_runs():
        # Check that multiple runs advance the clock as expected
        group = NeuronGroup(5, 'v : volt')
        mon = StateMonitor(group, 'v', record=True)
        run(1 * ms)
        run(1 * ms)
        device.build(direct_call=False, **device.build_options)
        assert_equal(defaultclock.t, 2 * ms)
        assert_equal(mon.t[0], 0 * ms)
        assert_equal(mon.t[-1], 2 * ms - defaultclock.dt)


Tests can also be written specifically for a standalone device (they then have
to include the `~brian2.devices.device.set_device` call and possibly the
`~brian2.devices.device.Device.build` call explicitly). In this case tests
have to be annotated with the name of the device (e.g. ``'cpp_standalone'``)
and with ``'standalone_only'`` to exclude this test from the runtime tests.
Such code would look like this for a single `run` call, i.e. using the automatic
"build on run" feature::

    import pytest
    from brian2 import *

    @pytest.mark.cpp_standalone
    @pytest.mark.standalone_only
    def test_cpp_standalone():
        set_device('cpp_standalone', directory=None)
        # set up simulation
        # run simulation
        run(...)
        # check simulation results


If the code uses more than one `run` statement, it needs an explicit
`~brian2.devices.device.Device.build` call::

    import pytest
    from brian2 import *

    @pytest.mark.cpp_standalone
    @pytest.mark.standalone_only
    def test_cpp_standalone():
        set_device('cpp_standalone', build_on_run=False)
        # set up simulation
        # run simulation
        run(...)
        # do something
        # run again
        run(...)
        device.build(directory=None)
        # check simulation results


Summary
^^^^^^^
+------------------------------------------+------------------------+-------------------------------------------------------------+
| ``@pytest.mark`` marker                  | Executed for devices   | explicit use of `device`                                    |
+==========================================+========================+=============================================================+
| ``codegen_independent``                  | independent of devices | *none*                                                      |
+------------------------------------------+------------------------+-------------------------------------------------------------+
| *none*                                   | Runtime targets        | *none*                                                      |
+------------------------------------------+------------------------+-------------------------------------------------------------+
| ``standalone_compatible``                | Runtime and standalone | *none*                                                      |
+------------------------------------------+------------------------+-------------------------------------------------------------+
| ``standalone_compatible, multiple_runs`` | Runtime and standalone | ``device.build(direct_call=False, **device.build_options)`` |
+------------------------------------------+------------------------+-------------------------------------------------------------+
| ``cpp_standalone, standalone_only``      | C++ standalone device  | ``set_device('cpp_standalone')``                            |
|                                          |                        | ``...``                                                     |
|                                          |                        | ``device.build(directory=None)``                            |
+------------------------------------------+------------------------+-------------------------------------------------------------+
| ``my_device, standalone_only``           | "My device"            | ``set_device('my_device')``                                 |
|                                          |                        | ``...``                                                     |
|                                          |                        | ``device.build(directory=None)``                            |
+------------------------------------------+------------------------+-------------------------------------------------------------+

Doctests
~~~~~~~~
Doctests are executable documentation. In the ``Examples`` block of a class or
function documentation, simply write code copied from an interactive Python
session (to do this from ipython, use ``%doctestmode``), e.g.::

    >>> from brian2.utils.stringtools import word_substitute
    >>> expr = 'a*_b+c5+8+f(A)'
    >>> print(word_substitute(expr, {'a':'banana', 'f':'func'}))
    banana*_b+c5+8+func(A)

During testing, the actual output will be compared to the expected output and
an error will be raised if they don't match. Note that this comparison is
strict, e.g. trailing whitespace is not ignored. There are various ways of
working around some problems that arise because of this expected exactness (e.g.
the stacktrace of a raised exception will never be identical because it contains
file names), see the `doctest documentation`_ for details.

Doctests can (and should) not only be used in docstrings, but also in the
hand-written documentation, making sure that the examples actually work. To
turn a code example into a doc test, use the ``.. doctest::`` directive, see
:doc:`/user/equations` for examples written as doctests. For all doctests,
everything that is available after ``from brian2 import *`` can be used
directly. For everything else, add import statements to the doctest code or --
if you do not want the import statements to appear in the document -- add them
in a ``.. testsetup::`` block. See the documentation for
`Sphinx's doctest extension`_ for more details.

Doctests are a great way of testing things as they not only make sure that the
code does what it is supposed to do but also that the documentation is up to
date!

.. _`doctest documentation`: https://docs.python.org/2/library/doctest.html
.. _`Sphinx's doctest extension`: http://www.sphinx-doc.org/en/stable/ext/doctest.html


Correctness tests
~~~~~~~~~~~~~~~~~
[These do not exist yet for brian2]. Unit tests test a specific function or
feature in isolation. In addition, we want to have tests where a complex piece
of code (e.g. a complete simulation) is tested. Even if it is sometimes
impossible to really check whether the result is correct (e.g. in the case of
the spiking activity of a complex network), a useful check is also whether the
result is *consistent*. For example, the spiking activity should be the same
when using code generation for Python or C++. Or, a network could be pickled
before running and then the result of the run could be compared to a second run
that starts from the unpickled network.
