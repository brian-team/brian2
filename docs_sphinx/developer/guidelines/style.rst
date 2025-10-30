Coding conventions
==================
General recommendations
-----------------------
Syntax is chosen as much as possible from the user point of view,
to reflect the concepts as directly as possible. Ideally, a Brian script
should be readable by someone who doesn't know Python or Brian, although this
isn't always possible. Function, class and keyword argument names should be
explicit rather than abbreviated and consistent across Brian. See Romain's paper
`On the design of script languages for neural simulators
<http://briansimulator.org/WordPress/wp-content/uploads/2012/05/On-the-design-of-script-languages-for-neural-simulation.pdf>`__
for a discussion.

.. _code_style:

Code style
~~~~~~~~~~

We use the `PEP-8 coding conventions <https://www.python.org/dev/peps/pep-0008/>`__
for our code, and use the `black formatting tool <https://black.readthedocs.io>`__ to
enforce a consistent code style. To make sure your code is formatted in the same way,
you can either `integrate black with your editor/IDE <https://black.readthedocs.io/en/stable/integrations/editors.html>`__,
or install the ``pre-commit`` tool
(`pre-commit documentation <https://pre-commit.com/>`__), and install it as a "git hook"
with ``pre-commit install``. This will automatically run ``black`` (and in the future,
additional linting tools) before each commit. In case that run changes the code
formatting, it will reformat the relevant files, and you will have to ``git add`` these
changes before doing the commit. For pull requests, this check will also be run
automatically on the GitHub CI infrastructure.

.. note::

    In rare cases like manually aligned tables of related values, you can use
    ``fmt: skip`` (for single lines), or ``fmt: off`` and ``fmt: on`` (for code blocks),
    to exclude code from black's formatting. Also note that the code formatting is only
    enforced for files in the ``brian2`` package itself, code examples in the
    documentation or examples and tutorials do not have to follow black's style.

The code style includes the following conventions in particular:

* Use 4 spaces instead of tabs per indentation level
* Use spaces after commas and around the following binary operators:
  assignment (=), augmented assignment (+=, -= etc.),
  comparisons (==, <, >, !=, <>, <=, >=, in, not in, is, is not),
  Booleans (and, or, not).
* Do *not* use spaces around the equals sign in keyword arguments or when
  specifying default values. Neither put spaces immediately inside parentheses,
  brackets or braces, immediately before the open parenthesis that starts the
  argument list of a function call, or immediately before the open parenthesis
  that starts an indexing or slicing.
* Avoid using a backslash for continuing lines whenever possible, instead use
  Python's implicit line joining inside parentheses, brackets and braces.

Imports
~~~~~~~
Imports should be on different lines (e.g. do not use ``import sys, os``) and should be
grouped in the following order, using blank lines between each group:

  	1. standard library imports
  	2. third-party library imports (e.g. numpy, scipy, sympy, ...)
  	3. brian imports

This rule is enforced by using the `isort <https://pycqa.github.io/isort/>`__ tool,
which is integrated with ``pre-commit`` in the same way as ``black``, described above.

.. note::

    In rare cases, where logical grouping makes more sense that ``isort``'s ordering,
    or when the order of imports matters, you can skip sorting imports in a file by
    including the comment ``# isort:skip_file``.

Additional rules for imports:

* Use absolute imports for everything outside of "your" package, e.g. if you
  are working in `brian2.equations`, import functions from the stringtools
  modules via ``from brian2.utils.stringtools import ...``. Use the full path
  when importing, e.g. do ``from brian2.units.fundamentalunits import seconds``
  instead of ``from brian2 import seconds``.
* Use "new-style" relative imports for everything in "your" package, e.g. in
  ``brian2.codegen.functions.py`` import the `Function` class as
  ``from .specifiers import Function``.
* Do not use wildcard imports (``from brian2 import *``), instead import only the
  identifiers you need, e.g. ``from brian2 import NeuronGroup, Synapses``. For
  packages like numpy that are used a lot, use ``import numpy as np``. But
  note that the user should still be able to do something like
  ``from brian2 import *`` (and this style can also be freely used in examples
  and tests, for example). Modules always have to use the ``__all__`` mechanism
  to specify what is being made available with a wildcard import. As an
  exception from this rule, the main ``brian2/__init__.py`` may use wildcard
  imports.

String formatting
-----------------
In general, we use Python `f-strings <https://docs.python.org/3/reference/lexical_analysis.html#formatted-string-literals>`__
instead of the ``.format`` method or the `%` operator to format strings. For example, rather use::

    raise KeyError(f"Unknown variable '{var}'")  # ✔

instead of::

    raise KeyError("Unknown variable '{}'".format(var))  #  ❌
    raise KeyError("Unknown variable %s" % var)  #  ❌

There are some corner cases where it still makes sense to use either of these, though.
The `~str.format` method can be useful when processing several strings instead of single literals::

    formatted = []
    for s in strings:
        formatted.append(s.format(**values))

The `%` operator, or string concatenation, can be used when dealing with strings that contain curly braces, which would
become difficult to read as an f-string::

    latex_code = r'\begin{equation}%s\end{equation}' % equation  # OK
    latex_code = r'\begin{equation}' + equation + r'\end{equation}' # OK

Python does not make a difference between single quotation marks and double quotation marks. For consistency,
we use black's style that double quotes (i.e. `"..."`) are used everywhere, except when
this would lead to escaping of double quotation marks within the string itself
(e.g. for ``include = f'#include "{header_file}"'``). When you need to display text to
the user (e.g. in exception messages), use single quotation marks to highlight words to
avoid this situation (e.g. `"The 'item' value should not be 0."`).

Commits only changing the style
-------------------------------
We have sometimes made big commits updating the style in our code, which can make using tools like ``git blame`` more
difficult, since many lines are affected by such commits. We add the references to such commits to a file
``.git-blame-ignore-revs`` in the main directory, and you can tell ``git blame`` to ignore these commits with::

    git config blame.ignoreRevsFile .git-blame-ignore-revs
