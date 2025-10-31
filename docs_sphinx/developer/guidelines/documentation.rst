Documentation
=============

It is very important to maintain documentation. We use the
`Sphinx documentation generator <http://www.sphinx-doc.org/en/stable/>`__
tools. The documentation is all hand written. Sphinx source files are stored in the
``docs_sphinx`` folder. The HTML files can be generated via the script
``dev/tools/docs/build_html_brian2.py`` and end
up in the ``docs`` folder.

Most of the documentation is stored directly in the Sphinx
source text files, but reference documentation for important Brian classes and
functions are kept in the documentation strings of those classes themselves.
This is automatically pulled from these classes for the reference manual
section of the documentation. The idea is to keep the definitive reference
documentation near the code that it documents, serving as both a comment for
the code itself, and to keep the documentation up to date with the code.

The reference documentation includes all classes, functions and other objects
that are defined in the modules and only documents them in the module where
they were defined. This makes it possible to document a class like
`~brian2.units.fundamentalunits.Quantity` only in `brian2.units.fundamentalunits`
and not additionally in `brian2.units` and `brian2`. This mechanism relies on
the ``__module__`` attribute, in some cases, in particular when wrapping a
function with a decorator (e.g. `~brian2.units.fundamentalunits.check_units`),
this attribute has to be set manually::

	foo.__module__ = __name__

Without this manual setting, the function might not be documented at all or in
the wrong module.

In addition to the reference, all the examples in the examples folder are
automatically included in the documentation.

Note that you can directly link to github issues using ``:issue:`issue number```, e.g.
writing ``:issue:`33``` links to a github issue about running benchmarks for Brian 2:
:issue:`33`. This feature should rarely be used in the main documentation, reserve its
use for release notes and important known bugs.

Docstrings
----------

Every module, class, method or function has to start with a docstring, unless
it is a private or special method (i.e. starting with ``_`` or ``__``) *and* it
is obvious what it does. For example, there is normally no need to document
``__str__`` with "Return a string representation.".

For the docstring format, we use the our own sphinx extension (in
`brian2/sphinxext`) based on
`numpydoc <https://pypi.python.org/pypi/numpydoc/>`__, allowing to write
docstrings that are well readable both in sourcecode as well as in the
rendered HTML. We generally follow the `format used by numpy
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`__

When the docstring uses variable, class or function names, these should be
enclosed in single backticks. Class and function/method names will be
automatically linked to the corresponding documentation. For classes imported
in the main brian2 package, you do not have to add the package name, e.g.
writing ```NeuronGroup``` is enough. For other classes, you have to give the
full path, e.g. ```brian2.units.fundamentalunits.UnitRegistry```. If it is
clear from the context where the class is (e.g. within the documentation of
`~brian2.units.fundamentalunits.UnitRegistry`), consider using the ``~``
abbreviation: ```~brian2.units.fundamentalunits.UnitRegistry``` displays only
the class name: `~brian2.units.fundamentalunits.UnitRegistry`. Note that you do
not have to enclose the exception name in a "Raises" or "Warns" section, or
the class/method/function name in a "See Also" section in backticks, they will
be automatically linked (putting backticks there will lead to incorrect display
or an error message),

Inline source fragments should be enclosed in  double backticks.

Class docstrings follow the same conventions as method docstrings and should
document the ``__init__`` method, the ``__init__`` method itself does not need
a docstring.

Documenting functions and methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The docstring for a function/method should start with a one-line description of
what the function does, without referring to the function name or the names of
variables. Use a "command style" for this summary, e.g. "Return the result."
instead of "Returns the result." If the signature of the function cannot be
automatically extracted because of an decorator (e.g. `check_units`), place a
signature in the very first row of the docstring, before the one-line
description.

For methods, do not document the ``self`` parameter, nor give information about
the method being static or a class method (this information will be
automatically added to the documentation).

Documenting classes
~~~~~~~~~~~~~~~~~~~
Class docstrings should use the same "Parameters" and "Returns" sections as
method and function docstrings for documenting the ``__init__`` constructor. If
a class docstring does not have any "Attributes" or "Methods" section, these
sections will be automatically generated with all documented (i.e. having a
docstring), public (i.e. not starting with `_`) attributes respectively methods
of the class. Alternatively, you can provide these sections manually. This is
useful for example in the `Quantity` class, which would otherwise include the
documentation of many `ndarray` methods, or when you want to include
documentation for functions like ``__getitem__`` which would otherwise not be
documented. When specifying these sections, you only have to state the names of
documented methods/attributes but you can also provide direct documentation.
For example::

    Attributes
    ----------
    foo
    bar
    baz
        This is a description.

This can be used for example for class or instance attributes which do not
have "classical" docstrings. However, you can also use a special syntax: When
defining class attributes in the class body or instance attributes in
``__init__`` you can use the following variants (here shown for instance
attributes)::

    def __init__(a, b, c):
        #: The docstring for the instance attribute a.
        #: Can also span multiple lines
        self.a = a

        self.b = b #: The docstring for self.b (only one line).

        self.c = c
        'The docstring for self.c, directly *after* its definition'

Long example of a function docstring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a very long docstring, showing all the possible sections. Most of the
time no See Also, Notes or References section is needed::

    def foo(var1, var2, long_var_name='hi') :
    """
    A one-line summary that does not use variable names or the function name.

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var1`.

    Parameters
    ----------
    var1 : array_like
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    var2 : int
        The type above can either refer to an actual Python type
        (e.g. ``int``), or describe the type of the variable in more
        detail, e.g. ``(N,) ndarray`` or ``array_like``.
    Long_variable_name : {'hi', 'ho'}, optional
        Choices in brackets, default first when optional.

    Returns
    -------
    describe : type
        Explanation
    output : type
        Explanation
    tuple : type
        Explanation
    items : type
        even more explaining

    Raises
    ------
    BadException
        Because you shouldn't have done that.

    See Also
    --------
    otherfunc : relationship (optional)
    newfunc : Relationship (optional), which could be fairly long, in which
              case the line wraps here.
    thirdfunc, fourthfunc, fifthfunc

    Notes
    -----
    Notes about the implementation algorithm (if needed).

    This can have multiple paragraphs.

    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    And even use a greek symbol like :math:`omega` inline.

    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.

    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a=[1,2,3]
    >>> print([x + 3 for x in a])
    [4, 5, 6]
    >>> print("a\nb")
    a
    b

    """

    pass
