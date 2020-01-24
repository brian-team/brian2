Brian 2 documentation
=====================

Brian is a simulator for spiking neural networks. It is written in the Python
programming language and is available on almost all platforms. We believe
that a simulator should not only save the time of processors, but also the
time of scientists. Brian is therefore designed to be easy to learn and use,
highly flexible and easily extensible.

To get an idea of what writing a simulation in Brian looks like, take a look
at :doc:`a simple example </examples/CUBA>`, or run our
`interactive demo <http://mybinder.org/v2/gh/brian-team/brian2-binder/master?filepath=demo.ipynb>`_.

.. only:: html

    You can actually edit and run the examples in the browser without having to
    install Brian, using the Binder service (note: sometimes this service is down
    or running slowly):

    .. image:: http://mybinder.org/badge.svg
        :target: http://mybinder.org/v2/gh/brian-team/brian2-binder/master?filepath=demo.ipynb

Once you have a feel for what is involved in using Brian, we recommend you
start by following the
:doc:`installation instructions </introduction/install>`, and in case you are
new to the Python programming language, having a look at
:doc:`introduction/scripts`. Then, go through the
:doc:`tutorials </resources/tutorials/index>`, and finally
read the :doc:`User Guide </user/index>`.

While reading the documentation, you will see the names of certain functions
and classes are highlighted links (e.g. `PoissonGroup`). Clicking on these
will take you to the "reference documentation". This section is automatically
generated from the code, and includes complete and very detailed information,
so for new users we recommend sticking to the :doc:`../user/index`. However,
there is one feature that may be useful for all users. If you click on,
for example, `PoissonGroup`, and scroll down to the bottom, you'll get a
list of all the example code that uses `PoissonGroup`. This is available
for each class or method, and can be helpful in understanding how a
feature works.

Finally, if you're having problems, please do let us know at our
:doc:`support page </introduction/support>`.

Please note that all interactions (e.g. via the mailing list or on github) should adhere
to our :doc:`Code of Conduct <introduction/code_of_conduct>`.

Contents:

.. toctree::
   :maxdepth: 2
   :titlesonly:

   introduction/index
   resources/tutorials/index
   user/index
   advanced/index

.. toctree::
   :maxdepth: 1
   :titlesonly:

   examples/index
   Reference documentation <reference/brian2>
   developer/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
