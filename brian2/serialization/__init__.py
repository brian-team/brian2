"""
brian2.serialization
~~~~~~~~~~~~~~~~~~~~

Portable serialization of Brian2 Network state.

Provides :func:`serialize_network_state` and :func:`restore_network_state`
which store / load network state as a ``.brian`` ZIP archive (JSON + npz)
instead of Python pickle.

See :mod:`brian2.serialization.brian_format` for the full specification.
"""

from .brian_format import restore_network_state, serialize_network_state

__all__ = ["serialize_network_state", "restore_network_state"]
