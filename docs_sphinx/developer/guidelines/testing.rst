Testing
=======

Brian uses the `nose package <https://nose.readthedocs.org>`__
for its testing framework. To check the code coverage of the test suite, we use 
`coverage.py <http://nedbatchelder.com/code/coverage>`__. 

Running the test suite
----------------------
The nosetests tool automatically finds tests in the code. When brian2 is in your
Python path or when you are in the main brian2 directory, you can start the test
suite with::

	$ nosetests brian2 --with-doctest

This should show no errors or failures but possibly a number of skipped tests.
The recommended way however is to import brian2 and call the test function,
which gives you convenient control over which tests are run::

	>>> import brian2
	>>> brian2.test() 

By default, this runs the test suite for all available (runtime) code generation
targets. If you only want to test a specific target, provide it as an argument::

    >>> brian2.test('numpy')

If you want to test several targets, use a list of targets::

    >>> brian2.test(['weave', 'cython'])


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

    >>> brian2.test([], test_standalone='cpp_standalone')


Checking the code coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~
To check the code coverage under Linux (with coverage and nosetests in your
path) and generate a report, use the following commands (this assumes the
source code of Brian with the file ``.coveragerc`` in the directory
/path/to/brian)::

	$ coverage run --rcfile=/path/to/brian/.coveragerc $(which nosetests) --with-doctest brian2
	$ coverage report

Using ``coverage html`` you can also generate a HTML report which will end up
in the directory ``htmlcov``.


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
errors are raised when expected. For that you can use the `assert_raises`
function (also in `numpy.testing.utils`) which takes an Exception type and
a callable as arguments::

	assert_raises(DimensionMismatchError, lambda: 3*volt + 5*second)

Note that you cannot simply write ``3*volt + 5*second`` in the above example,
this would raise an exception before calling assert_raises. Using a callable
like the simple lambda expression above makes it possible for `assert_raises`
to catch the error and compare it against the expected type. You can also check
whether expected warnings are raised, see the documentation of the :doc:`logging
mechanism <logging>` for details

For simple functions, doctests (see below) are a great alternative to writing
classical unit tests.

By default, all tests are executed for all selected code generation targets
(see `Running the test suite`_ above). This is not useful for all tests, some
basic tests that for example test equation syntax or the use of physical units
do not depend on code generation and need therefore not to be repeated. To
execute such tests only once, they can be annotated with a
``codegen-independent`` attribute, using the `~nose.plugins.attrib.attr`
decorator::

    from nose.plugins.attrib import attr
    from brian2 import NeuronGroup

    @attr('codegen-independent')
    def test_simple():
        # Test that the length of a NeuronGroup is correct
        group = NeuronGroup(5, '')
        assert len(group) == 5

Tests that are not "codegen-independent" are by default only executed for the
runtimes device, i.e. not for the ``cpp_standalone`` device, for example.
However, many of those tests follow a common pattern that is compatible with
standalone devices as well: they set up a network, run it, and check the state
of the network afterwards. Such tests can be marked as
``standalone-compatible``, using the `~nose.plugins.attrib.attr` decorator in
the same way as for ``codegen-independent`` tests. Since standalone devices
usually have an internal state where they store information about arrays,
array assignments, etc., they need to be reinitialized after such a test. For
that use the `~nose.with_setup` decorator and provide the
`~brian2.devices.device.restore_device` function as the ``teardown``
argument::

    from nose import with_setup
    from nose.plugins.attrib import attr
    from numpy.testing.utils import assert_equal
    from brian2 import *
    from brian2.devices.device import restore_device

    @attr('standalone-compatible')
    @with_setup(teardown=restore_initial_state)
    def test_simple_run():
        # Check that parameter values of a neuron don't change after a run
        group = NeuronGroup(5, 'v : volt')
        group.v = 'i*mV'
        run(1*ms)
        assert_equal(group.v[:], np.arange(5)*mV)

As a rule of thumb:

* If a test does not have a `~brian2.core.network.Network.run` call, mark it as
  ``codegen-independent``
* If a test has only a single `~brian2.core.network.Network.run` and only reads state variable
  values after the run, mark it as ``standalone-compatible`` and register the
  `~brian2.devices.device.restore_device` teardown function

Tests can also be written specifically for a standalone device (they then have
to include the `~brian2.devices.device.set_device` and
`~brian2.devices.device.Device.build` calls explicitly). In this case tests
have to be annotated with the name of the device (e.g. ``'cpp_standalone'``)
and with ``'standalone-only'`` to exclude this test from the runtime tests.
Also, the device should be restored in the end::

    from nose import with_setup
    from nose.plugins.attrib import attr
    from brian2 import *
    from brian2.devices.device import restore_device

    @attr('cpp_standalone', 'standalone-only')
    @with_setup(teardown=restore_initial_state)
    def test_cpp_standalone():
        set_device('cpp_standalone')
        # set up simulation
        # run simulation
        device.build(...)
        # check simulation results

Doctests
~~~~~~~~
Doctests are executable documentation. In the ``Examples`` block of a class or
function documentation, simply write code copied from an interactive Python
session (to do this from ipython, use ``%doctestmode``), e.g.::

    >>> expr = 'a*_b+c5+8+f(A)'
    >>> print word_substitute(expr, {'a':'banana', 'f':'func'})
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

.. _`doctest documentation`: http://docs.python.org/2/library/doctest.html
.. _`Sphinx's doctest extension`: http://sphinx-doc.org/ext/doctest.html

Test attributes
~~~~~~~~~~~~~~~

As explained above, the test suite can be run with different subsets of the
available tests. For this, tests have to be annotated with the ``attr``
decorator available from ``nose.plugins.attrib``. Currently, the following
attributes are understood:

* **standalone**: A C++ standalone test (not run by default when calling ``brian2.test()``)
* **codegen-independent**: A test that does not use any code generation (run by default)
* **long**: A test that takes a long time to run (not run by default)

Attributes can be simply given as a string argument to the ``attr`` decorator:

.. code-block:: python
   :emphasize-lines: 3

    from nose.plugins.attrib import attr

    @attr('standalone')
    test_for_standalone():
        pass  # ...

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