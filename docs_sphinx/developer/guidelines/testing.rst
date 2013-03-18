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
Alternatively you can import brian2 and call the test function, e.g. in an
interactive Python session::

	>>> import brian2
	>>> brian2.test() 

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