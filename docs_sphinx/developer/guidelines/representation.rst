Representing Brian objects
=============================

``__repr__`` and ``__str__``
----------------------------

Every class should specify or inherit useful ``__repr__`` and ``__str__`` methods. The ``__repr__``
method should give the "official" representation of the object; if possible, this should be a valid
Python expression, ideally allowing for ``eval(repr(x)) == x``. The ``__str__`` method on the other
hand, gives an "informal" representation of the object. This can be anything that is helpful but
does not have to be Python code. For example:

.. doctest::

    >>> import numpy as np
    >>> ar = np.array([1, 2, 3]) * mV
    >>> print(ar)  # uses __str__
    [ 1.  2.  3.] mV
    >>> ar  # uses __repr__
    array([ 1.,  2.,  3.]) * mvolt

If the representation returned by ``__repr__`` is not Python code, it should be enclosed in
``<...>``, e.g. a `Synapses` representation might be ``<Synapses object with 64 synapses>``.

If you don't want to make the distinction between ``__repr__`` and ``__str__``, simply define only
a ``__repr__`` function, it will be used instead of ``__str__`` automatically (no need to write
``__str__ = __repr__``). Finally, if you include the class name in the representation (which you
should in most cases), use ``self.__class__.__name__`` instead of spelling out the name explicitly
-- this way it will automatically work correctly for subclasses. It will also prevent you from
forgetting to update the class name in the representation if you decide to rename the class.

LaTeX representations with sympy
--------------------------------
Brian objects dealing with mathematical expressions and equations often internally use sympy.
Sympy's `~sympy.printing.latex.latex` function does a nice job of converting expressions into
LaTeX code, using fractions, root symbols, etc. as well as converting greek variable names into
corresponding symbols and handling sub- and superscripts. For the conversion of variable names
to work, they should use an underscore for subscripts and two underscores for superscripts::

    >>> from sympy import latex, Symbol
    >>> tau_1__e = Symbol('tau_1__e')
    >>> print(latex(tau_1__e))
    \tau^{e}_{1}

Sympy's printer supports formatting arbitrary objects, all they have to do is to implement a
``_latex`` method (no trailing underscore). For most Brian objects, this is unnecessary as they will
never be formatted with sympy's LaTeX printer. For some core objects, in particular the units,
is is useful, however, as it can be reused in LaTeX representations for ipython (see below).
Note that the ``_latex`` method should not return ``$`` or ``\begin{equation}`` (sympy's method
includes a ``mode`` argument that wraps the output automatically).

Representations for ipython
---------------------------------

"Old" ipython console
~~~~~~~~~~~~~~~~~~~~~

In particular for representations involing arrays or lists, it can be useful to break up the
representation into chunks, or indent parts of the representation. This is supported by the
ipython console's "pretty printer". To make this work for a class, add a
``_repr_pretty_(self, p, cycle)`` (note the *single* underscores) method. You can find more
information in the `ipython documentation <http://ipython.org/ipython-doc/dev/api/generated/IPython.lib.pretty.html#extending>`__ .

"New" ipython console (qtconsole and notebook)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new ipython consoles, the qtconsole and the ipython notebook support a much richer set of
representations for objects. As Brian deals a lot with mathematical objects, in particular the
LaTeX and to a lesser extent the HTML formatting capabilities of the ipython notebook are
interesting. To support LaTeX representation, implement a `_repr_latex_` method returning the
LaTeX code (*including* ``$``, ``\begin{equation}`` or similar). If the object already has a
``_latex`` method (see `LaTeX representations with sympy`_ above), this can be as simple as::

    def _repr_latex_(self):
        return sympy.latex(self, mode='inline')  # wraps the expression in $ .. $

The LaTeX rendering only supports a single mathematical block. For complex objects, e.g.
`NeuronGroup` it might be useful to have a richer representation. This can be achieved by returning
HTML code from ``_repr_html_`` -- this HTML code is processed by MathJax so it can include literal
LaTeX code that will be transformed before it is rendered as HTML. An object containing two
equations could therefore be represented with a method like this::

    def _repr_html_(self):
        return '''
        <h3> Equation 1 </h3>
        {eq_1}
        <h3> Equation 2 </h3>
        {eq_2}'''.format(eq_1=sympy.latex(self.eq_1, mode='equation'),
                         eq_2=sympy.latex(self.eq_2, mode='equation'))
