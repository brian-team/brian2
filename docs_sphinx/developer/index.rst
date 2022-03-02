Developer's guide
=================

This section is intended as a guide to how Brian functions internally for
people developing Brian itself, or extensions to Brian. It may also be of some
interest to others wishing to better understand how Brian works internally.

If you use `VS code <https://code.visualstudio.com/>`_ as your development
environment, it will offer to automatically build a Brian development
`Docker <https://www.docker.com/>`_ container when you open the repository,
with all the required dependencies installed and configured. Further
`documentation <https://github.com/brian-team/brian2/blob/master/.devcontainer/README.md>`_
for this approach can be found in the ``.devcontainer`` directory.

.. toctree::
   :maxdepth: 2

   guidelines/index
   units
   equations_namespaces
   variables_indices
   preferences
   functions
   codegen
   devices
   openmp
   GSL
