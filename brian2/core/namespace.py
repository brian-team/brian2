'''
Implementation of the namespace system, used to resolve the identifiers in
model equations of `NeuronGroup` and `Synapses`
'''
import inspect
import itertools
import numbers
import weakref

try:
    from collections import OrderedDict
except ImportError:
    # OrderedDict was added in Python 2.7, use backport for Python 2.6
    from brian2.utils.ordereddict import OrderedDict

import numpy as np

from brian2.utils.logger import get_logger
from brian2.units.fundamentalunits import Quantity, standard_unit_register
from brian2.units.stdunits import stdunits
from brian2.core.functions import DEFAULT_FUNCTIONS

from .functions import Function

__all__ = ['resolve',
           'resolve_all',
           'get_local_namespace',
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


def _conflict_warning(message, resolutions):
    '''
    A little helper functions to generate warnings for logging. Specific
    to the `Namespace.resolve` method and should only be used by it.
    
    Parameters
    ----------
    message : str
        The first part of the warning message.
    resolutions : list of str
        A list of (namespace, object) tuples.
    '''
    if len(resolutions) == 0:
        # nothing to warn about
        return
    elif len(resolutions) == 1:
        second_part = ('but also refers to a variable in the %s namespace:'
                       ' %r') % (resolutions[0][0], resolutions[0][1])
    else:
        second_part = ('but also refers to a variable in the following '
                       'namespaces: %s') % (', '.join([r[0] for r in resolutions]))

    logger.warn(message + ' ' + second_part,
                'Namespace.resolve.resolution_conflict', once=True)


def _same_function(func1, func2):
    '''
    Helper function, used during namespace resolution for comparing whether to
    functions are the same. This takes care of treating a function and a
    `Function` variables whose `Function.pyfunc` attribute matches as the
    same. This prevents the user from getting spurious warnings when having
    for example a numpy function such as :np:func:`~random.randn` in the local
    namespace, while the ``randn`` symbol in the numpy namespace used for the
    code objects refers to a `RandnFunction` specifier.
    '''
    # use the function itself if it doesn't have a pyfunc attribute and try
    # to create a weak proxy to make a comparison to other weak proxys return
    # true
    func1 = getattr(func1, 'pyfunc', func1)
    try:
        func1 = weakref.proxy(func1)
    except TypeError:
        pass  # already a weakref proxy
    func2 = getattr(func2, 'pyfunc', func2)
    try:
        func2 = weakref.proxy(func2)
    except TypeError:
        pass

    return func1 is func2  

    
def resolve(identifier, group, run_namespace=None, level=0,
            strip_units=False):
    '''
    Resolve an external identifier in the context of a `Group`. If the `Group`
    declares an explicit namespace, this namespace is used in addition to the
    standard namespace for units and functions. Additionally, the namespace in
    the `run_namespace` argument (i.e. the namespace provided to `Network.run`)
    is used. Only if both are undefined, the implicit namespace of
    surrounding variables in the stack frame where the original call was made
    is used (to determine this stack frame, the `level` argument has to be set
    correctly).

    Parameters
    ----------
    identifier : str
        The name to resolve.
    group : `Group`
        The group that potentially defines an explicit namespace for looking up
        external names.
    run_namespace : dict, optional
        A namespace (mapping from strings to objects), as provided as an
        argument to the `Network.run` function.
    level : int, optional
        How far to go up in the stack to find the calling frame.
    strip_unit : bool, optional
        Whether to get rid of units (for the code used in simulation) or not
        (when checking the units of an expression, for example). Defaults to
        ``False``.
    '''
    # We save tuples of (namespace description, referred object) to
    # give meaningful warnings in case of duplicate definitions
    matches = []

    namespaces = OrderedDict()
    namespaces['units'] = _get_default_unit_namespace()
    namespaces['functions'] = get_default_numpy_namespace()
    if getattr(group, 'namespace', None) is not None:
        namespaces['group-specific'] = group.namespace
    if run_namespace is not None:
        namespaces['run'] = run_namespace

    if getattr(group, 'namespace', None) is None and run_namespace is None:
        namespaces['implicit'] = get_local_namespace(level+1)

    for description, namespace in namespaces.iteritems():
        if identifier in namespace:
            matches.append((description, namespace[identifier]))

    if len(matches) == 0:
        # No match at all
        raise KeyError(('The identifier "%s" could not be resolved.') %
                       (identifier))
    elif len(matches) > 1:
        # Possibly, all matches refer to the same object
        first_obj = matches[0][1]
        found_mismatch = False
        for m in matches:
            if m[1] is first_obj:
                continue
            if _same_function(m[1], first_obj):
                continue
            try:
                proxy = weakref.proxy(first_obj)
                if m[1] is proxy:
                    continue
            except TypeError:
                pass

            # Found a mismatch
            found_mismatch = True
            break

        if found_mismatch:
            _conflict_warning(('The name "%s" refers to different objects '
                               'in different namespaces used for resolving. '
                               'Will use the object from the %s namespace '
                               'with the value %r') %
                              (identifier, matches[0][0],
                               first_obj), matches[1:])

    # use the first match (according to resolution order)
    resolved = matches[0][1]

    # Replace pure Python functions by a Functions object
    if callable(resolved) and not isinstance(resolved, Function):
        resolved = Function(resolved)

    return resolved


def resolve_all(identifiers, group, run_namespace=None, level=0,
            strip_units=False):
    resolutions = {}
    for identifier in identifiers:
        resolved = resolve(identifier, group, run_namespace=run_namespace,
                           level=level+1, strip_units=strip_units)
        resolutions[identifier] = resolved

    return resolutions


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
