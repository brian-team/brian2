"""
Implementation of the namespace system, used to resolve the identifiers in
model equations of `NeuronGroup` and `Synapses`
"""

import collections
import inspect
import itertools

from brian2.utils.logger import get_logger
from brian2.units.fundamentalunits import (
    standard_unit_register,
    additional_unit_register,
)
from brian2.units.stdunits import stdunits
from brian2.core.functions import DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS

__all__ = [
    "get_local_namespace",
    "DEFAULT_FUNCTIONS",
    "DEFAULT_UNITS",
    "DEFAULT_CONSTANTS",
]

logger = get_logger(__name__)


def get_local_namespace(level):
    """
    Get the surrounding namespace.

    Parameters
    ----------
    level : int, optional
        How far to go back to get the locals/globals. Each function/method
        call should add ``1`` to this argument, functions/method with a
        decorator have to add ``2``.

    Returns
    -------
    namespace : dict
        The locals and globals at the given depth of the stack frame.
    """
    # Get the locals and globals from the stack frame
    frame = inspect.currentframe()
    for _ in range(level + 1):
        frame = frame.f_back
    # We return the full stack here, even if it contains a lot of stuff we are
    # not interested in -- it is cheaper to later raise an error when we find
    # a specific object with an incorrect type instead of going through this big
    # list now to check the types of all objects
    return dict(itertools.chain(frame.f_globals.items(), frame.f_locals.items()))


def _get_default_unit_namespace():
    """
    Return the namespace that is used by default for looking up units when
    defining equations. Contains all registered units and everything from
    `brian2.units.stdunits` (ms, mV, nS, etc.).

    Returns
    -------
    namespace : dict
        The unit namespace
    """
    namespace = collections.OrderedDict(standard_unit_register.units)
    namespace.update(stdunits)
    # Include all "simple" units from additional_units, i.e. units like mliter
    # but not "newton * metre"
    namespace.update(
        dict(
            (name, unit)
            for name, unit in additional_unit_register.units.items()
            if not unit.iscompound
        )
    )
    return namespace


DEFAULT_UNITS = _get_default_unit_namespace()
