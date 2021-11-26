Support
=======

If you are stuck with a problem using Brian, please do get in touch at our
`community forum <http://brian.discourse.group>`__.

You can save time by following this procedure when reporting a problem:

 1.   Do try to solve the problem on your own first. Read the documentation,
      including using the search feature, index and reference documentation.
 2.   Search the mailing list archives to see if someone else already had the
      same problem.
 3.   Before writing, try to create a minimal example that reproduces the
      problem. You’ll get the fastest response if you can send just a handful
      of lines of code that show what isn’t working.

.. _which_version:

Which version of Brian am I using?
----------------------------------
When reporting problems, it is important to include the information what exact version
of Brian you are using. The different install methods listed in :doc:`install` provide
different mechanisms to get this information. For example, if you used ``conda`` for
installing Brian, you can use ``conda list brian2``; if you used ``pip``, you can use
``pip show brian2``.

A general method that works independent of the installation method is to ask the Brian
package itself:

.. code-block:: pycon

    >>> import brian2
    >>> print(brian2.__version__)  # doctest: +SKIP
    2.4.2

This method also has the advantage that you can easily call it from the same environment
(e.g. an IDE or a Jupyter Notebook) that you use when you execute Brian scripts. This
helps avoiding mistakes where you think you use a specific version but in fact you use a
different one. In such cases, it can also be helpful to look at Brian's ``__file__``
attribute:

.. code-block:: pycon

    >>> print(brian2.__file__) # doctest: +SKIP
    /home/marcel/anaconda3/envs/brian2_test/lib/python3.9/site-packages/brian2/__init__.py

In the above example, it shows that the ``brian2`` installation in the conda environment
``brian2_test`` is used.

If you installed a :ref:`development version <development_install>` of Brian, then the
version number will contain additional information:

.. code-block:: pycon

    >>> print(brian2.__version__) # doctest: +SKIP
    2.4.2.post0.dev408

The above means that the Brian version that is used has 408 additional commits that were
added after the 2.4.2 release. To get the exact git commit for the local Brian
installation, use:

.. code-block:: pycon

    >>> print(brian2.__git_revision__) # doctest: +SKIP
    d2cb4a85f804037ef055503975d822ff3f473ccf

To get more information about this commit, you can append it to the repository URL
on GitHub as ``/commit/<commit id>`` (where the first few characters of the
``<commit id>`` are enough), e.g. for the commit referenced above:
https://github.com/brian-team/brian2/commit/d2cb4a85
