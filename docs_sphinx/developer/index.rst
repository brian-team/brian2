Developer's guide
=================

This section is intended as a guide to how Brian functions internally for
people developing Brian itself, or extensions to Brian. It may also be of some
interest to others wishing to better understand how Brian works internally.

Setting up a development environment
-------------------------------------

Option 1: Devcontainers (Highly Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest and most reliable way to set up a development environment is to use
`VS Code Devcontainers <https://code.visualstudio.com/docs/devcontainers/containers>`_.
When you open the Brian repository in
`VS Code <https://code.visualstudio.com/>`_, it will offer to automatically
build a `Docker <https://www.docker.com/>`_ container with all the required
dependencies installed and configured — no manual setup needed.

For full details, see the
`Devcontainer README <https://github.com/brian-team/brian2/blob/master/.devcontainer/README.md>`_
in the ``.devcontainer`` directory.

Option 2: Local Setup using ``uv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer a local setup without Docker, you can use
`uv <https://docs.astral.sh/uv/>`_ to install Brian in development mode.
``uv`` resolves all dependencies directly from the existing ``pyproject.toml``
file, keeping the repository clean without needing external dependency files.

1. `Install uv <https://docs.astral.sh/uv/getting-started/installation/>`_ if
   you haven't already.
2. From the root of the repository, run:

   .. code-block:: bash

      uv add --dev --editable .[docs,test]

   This will install Brian along with all documentation and testing
   dependencies in an editable (development) mode.

.. toctree::
   :maxdepth: 2

   guidelines/index
   units
   equations_namespaces
   variables_indices
   preferences
   functions
   codegen
   standalone
   openmp
   devices
   GSL
