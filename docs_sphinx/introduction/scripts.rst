Running Brian scripts
=====================

Brian scripts are standard Python scripts, and can therefore be run in the same way.
For interactive, explorative work, you might want to run code in a
jupyter notebook or in an ipython shell; for running finished code, you might want
to execute scripts through the standard Python interpreter; finally, for working on
big projects spanning multiple files, a dedicated integrated development environment
for Python could be a good choice. We will briefly describe all these approaches and
how they relate to Brian's examples and tutorial that are part of this documentation.
Note that none of these approaches are specific to Brian, so you can also search for
more information in any of the resources listed on the
`Python website <https://www.python.org/about/gettingstarted/>`_.

.. contents::
    :local:
    :depth: 1

Jupyter notebook
----------------
    The Jupyter Notebook is an open-source web application that allows you to create
    and share documents that contain live code, equations, visualizations and narrative text.

    (from `jupyter.org <https://jupyter.org>`_)

Jupyter notebooks are a great tool to run Brian code interactively, and include
the results of the simulations, as well as additional explanatory text in a common
document. Such documents have the file ending ``.ipynb``, and in Brian we use this
format to store the :doc:`../resources/tutorials/index`. These files can be displayed by
github (see e.g. `the first Brian tutorial <https://github.com/brian-team/brian2/blob/master/tutorials/1-intro-to-brian-neurons.ipynb>`_),
but in this case you can only see them as a static website, not edit or execute any
of the code.

To make the full use of such notebooks, you have to run them using the
jupyter infrastructure. The easiest option is to use the free
`mybinder.org <https://mybinder.org>`_ web service, which allows you to try out
Brian without installing it on your own machine. Links to run the tutorials on this
infrastructure are provided as *"launch binder"* buttons on the
:doc:`../resources/tutorials/index` page, and also for each of the
:doc:`../examples/index` at the top of the respective page (e.g.
:doc:`../examples/COBAHH`). To run notebooks on your own machine, you need
an installation of the jupyter notebook software on your own machine, as well as
Brian itself (see the :doc:`install` instructions for details). To open an existing
notebook, you have to download it to your machine. For the Brian tutorials, you find
the necessary links on the :doc:`../resources/tutorials/index` page. When you have
downloaded/installed everything necessary, you can start the jupyter notebook from
the command line (using Terminal on OS X/Linux, Command Prompt on Windows)::

    jupyter notebook

this will open the "Notebook Dashboard" in your default browser, from which you can
either open an existing notebook or create a new one. In the notebook, you can then
execute individual "code cells" by pressing ``SHIFT+ENTER`` on your keyboard, or
by pressing the play button in the toolbar.

For more information, see the
`jupyter notebook documentation <https://jupyter-notebook.readthedocs.io>`_.


IPython shell
-------------
An alternative to using the jupyter notebook is to use the interactive Python shell
`IPython <https://ipython.readthedocs.io/>`_, which runs in the
Terminal/Command Prompt. You can use it to directly type Python code interactively
(each line will be executed as soon as you press ``ENTER``), or to run Python code
stored in a file. Such files typically have the file ending ``.py``. You can
either create it yourself in a text editor of your choice (e.g. by copying&pasting
code from one of the :doc:`../examples/index`), or by downloading such files from
places such as github (e.g. the `Brian examples <https://github.com/brian-team/brian2/tree/master/examples>`_),
or `ModelDB <https://senselab.med.yale.edu/modeldb/>`_. You can then run them from
within IPython via::

    %run filename.py


Python interpreter
------------------
The most basic way to run Python code is to run it through the standard Python
interpreter. While you can also use this interpreter interactively, it is much less
convenient to use than the IPython shell or the jupyter notebook described above.
However, if all you want to do is to run an existing Python script (e.g. one of
the Brian :doc:`../examples/index`), then you can do this by calling::

    python filename.py

in a Terminal/Command Prompt.

Integrated development environment (IDE)
----------------------------------------
Python is a widely used programming language, and is therefore support by a wide
range of integrated development environments (IDE). Such IDEs provide features
that are very convenient for developing complex projects, e.g. they integrate
text editor and interactive Python console, graphical debugging tools, etc.
Popular environments include `Spyder <https://www.spyder-ide.org/>`_,
`PyCharm <https://www.jetbrains.com/pycharm/>`_, and
`Visual Studio Code <https://code.visualstudio.com/>`_, for an extensive list
see the `Python wiki <https://wiki.python.org/moin/IntegratedDevelopmentEnvironments>`_.
