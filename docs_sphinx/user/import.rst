Importing Brian
===============

After installation, Brian is avaiable in the `brian2` package. By doing a
wildcard import from this package, i.e.::

    from brian2 import *

you will not only get access to the ``brian2`` classes and functions, but also
to everything in the ``pylab`` package, which includes the plotting functions
from matplotlib_ and everything included in numpy/scipy (e.g. functions such
as ``arange``, ``linspace``, etc.). This is the style used in the documentation
and in the examples but not in the Brian code itself (see
:doc:`../developer/guidelines/style`).

If you want to use a wildcard import from Brian, but don't want to import all
the additional symbols provided by ``pylab``, you can use::

    from brian2.only import *

Note that whenever you use something different from the most general
``from brian2 import *`` statement, you should be aware that Brian overwrites
some numpy functions with their unit-aware equivalents
(see :doc:`../developer/units`). If you combine multiple wildcard imports, the
Brian import should therefore be the last import. Similarly, you should not
import and call overwritten numpy functions directly, e.g. by using
``import numpy as np`` followed by ``np.sin`` since this will not use the
unit-aware versions. To make this easier, Brian provides a ``brian2.numpy_``
package that provides access to everything in numpy but overwrites certain
functions. If you prefer to use prefixed names, the recommended way of doing
the imports is therefore::

    import brian2.numpy_ as np
    import brian2.only as br2

Note that it is safe to use e.g. ``np.sin`` and ``numpy.sin`` after a
``from brian2 import *``.

.. _matplotlib: http://matplotlib.org/


.. _dependency_checks:

Dependency checks
-----------------
Brian will check the dependency versions during import and raise an error for
an outdated dependency. An outdated dependency does not necessarily mean that
Brian cannot be run with it, it only means that Brian is untested on that
version. If you want to force Brian to run despite the outdated dependency, set
the `core.outdated_dependency_error` preference to ``False``. Note that this
cannot be done in a script, since you do not have access to the preferences
before importing `brian2`. See :doc:`../advanced/preferences` for instructions
how to set preferences in a file.