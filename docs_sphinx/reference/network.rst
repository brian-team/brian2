.. currentmodule:: brian2

Network
=======

.. autoclass:: Network

.. index::
	single: magic

Magic
-----

The word 'magic' when used in Brian refers to the general idea that you can
`run` a `Network` without explicitly defining what objects are in that
`Network`. When `run` is called, all existing Brian objects are used for the
`Network`. Explicitly, there is a global variable `magic_network`, a
`MagicNetwork` object (a type of `Network` that automatically updates itself
when new objects are created or deleted), and `run` uses this `magic_network`.
Note that in order to avoid bugs, sometimes the `magic_network` will enter an
invalid state which means it cannot be run. In this case, a `MagicError`
error will be raised. See documentation in
`run` and `MagicNetwork` for more details on this. If it happens, usually the
best thing to do is to construct a `Network` object which explicitly lists
which objects should be included, and avoid the use of the magic system.
Alternatively, if the problem is just that there are still references to
Brian objects that are no longer used, you can call the `clear` function to
remove these objects, and this will resolve the problems.

Functions
~~~~~~~~~

.. autofunction:: run
.. autofunction:: reinit
.. autofunction:: stop

`MagicNetwork`
~~~~~~~~~~~~~~

.. autodata:: magic_network
.. autoclass:: MagicNetwork

`MagicError`
~~~~~~~~~~~~

.. autoclass:: MagicError
