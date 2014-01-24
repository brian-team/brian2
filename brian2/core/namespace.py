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
from brian2.units.fundamentalunits import Quantity, standard_unit_register
from brian2.units.stdunits import stdunits
from brian2.core.functions import DEFAULT_FUNCTIONS

from .functions import Function

__all__ = ['get_local_namespace',
           'get_default_numpy_namespace',
           'DEFAULT_UNIT_NAMESPACE']

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
            # If possible, add a weak reference
            try:
                v = weakref.proxy(v)
            except TypeError:
                pass
            namespace[k] = v
    del frame
    return namespace


def get_default_numpy_namespace():
    '''
    Get the namespace of numpy functions/variables that is recognized by
    default. The namespace includes the constants :np:attr:`pi`,
    :np:attr:`e` and :np:attr:`inf` and the following
    functions:
    :np:func:`abs`, :np:func:`arccos`, :np:func:`arcsin`, :np:func:`arctan`,
    :np:func:`arctanh`, :np:func:`ceil`, :np:func:`cos`, :np:func:`cosh`,
    :np:func:`exp`, :np:func:`floor`, :np:func:`log`, :np:func:`sin`,
    :np:func:`sinh`, :np:func:`sqrt`, :np:func:`tan`, :np:func:`tanh`  
    
    Returns
    -------
    namespace : dict
        A dictionary mapping function/variable names to numpy objects or
        their unitsafe Brian counterparts.
    '''        
    # numpy constants
    # TODO: Make them accesible to sympy as well, maybe introduce a similar
    #       system as for functions, e.g. C++ would use M_PI for pi?
    namespace = {'pi': np.pi, 'e': np.e, 'inf': np.inf}
    
    # The default numpy functions
    namespace.update(DEFAULT_FUNCTIONS)

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

DEFAULT_UNIT_NAMESPACE = _get_default_unit_namespace()
