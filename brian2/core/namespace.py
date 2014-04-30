'''
Implementation of the namespace system, used to resolve the identifiers in
model equations of `NeuronGroup` and `Synapses`
'''
import inspect
import itertools
import numbers
import weakref

import numpy as np

from brian2.utils.logger import get_logger
from brian2.units.fundamentalunits import standard_unit_register
from brian2.units.stdunits import stdunits
from brian2.core.functions import DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS

from .functions import Function

__all__ = ['get_local_namespace',
           'DEFAULT_FUNCTIONS',
           'DEFAULT_UNITS',
           'DEFAULT_CONSTANTS']

logger = get_logger(__name__)


def get_local_namespace(level):
    '''
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
    '''
    # Get the locals and globals from the stack frame
    namespace = dict()
    frame = inspect.stack()[level + 1][0]
    for k, v in itertools.chain(frame.f_globals.iteritems(),
                                frame.f_locals.iteritems()):
        # We are only interested in numbers and functions, not in
        # everything else (classes, modules, etc.)
        if (((isinstance(v, (numbers.Number, np.ndarray, np.number, Function))) or
            (inspect.isfunction(v) and
                 hasattr(v, '_arg_units') and
                 hasattr(v, '_return_unit'))) and
                not k.startswith('_')):
            namespace[k] = v
    del frame
    return namespace


def _get_default_unit_namespace():
    '''
    Return the namespace that is used by default for looking up units when
    defining equations. Contains all registered units and everything from
    `brian2.units.stdunits` (ms, mV, nS, etc.).
    
    Returns
    -------
    namespace : dict
        The unit namespace
    '''    
    namespace = dict([(u.name, u) for u in standard_unit_register.units])
    namespace.update(stdunits)
    return namespace

DEFAULT_UNITS = _get_default_unit_namespace()
