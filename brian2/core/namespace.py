'''
Implementation of the namespace system, used to resolve the identifiers in
model equations of `NeuronGroup` and `Synapses`
'''
import inspect
import collections
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
import brian2.equations.equations as equations

from .functions import Function

__all__ = ['create_namespace',
           'CompoundNamespace',
           'get_local_namespace',
           'get_default_numpy_namespace',
           'DEFAULT_UNIT_NAMESPACE']

logger = get_logger(__name__)


def get_local_namespace(level=0):
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
    frame = inspect.stack()[1 + level][0]
    namespace = dict(frame.f_globals)
    namespace.update(frame.f_locals)
    return namespace


def create_namespace(explicit_namespace=None):
    namespace = CompoundNamespace()
    
    # Functions and units take precedence, overwriting them would lead to
    # very confusing equations. In particular, the Equations objects does not
    # take the namespace into account when determining the units of equations
    # (the ": unit" part) -- so an overwritten unit would be ignored there but
    # taken into account in the equation itself.
    namespace.add_namespace('numpy', get_default_numpy_namespace())
    namespace.add_namespace('units', DEFAULT_UNIT_NAMESPACE)
    
    if explicit_namespace is not None:
        namespace.add_namespace('user-defined', explicit_namespace)            
    
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
    Helper function, used during namespace resolution for comparing wether to
    functions are the same. This takes care of treating a function and a
    `Function` variables whose `Function.pyfunc` attribute matches as the
    same. This prevents the user from getting spurious warnings when having
    for example a numpy function such as :np:func:`~random.randn` in the local
    namespace, while the ``randn`` symbol in the numpy namespace used for the
    code objects refers to a `RandnFunction` specifier.
    '''
    # use the function itself if it doesn't have a pyfunc attribute
    func1 = getattr(func1, 'pyfunc', func1)
    func2 = getattr(func2, 'pyfunc', func2)
    
    return func1 is func2  


class CompoundNamespace(collections.Mapping):

    def __init__(self):        
        self.namespaces = OrderedDict()        
    
    is_explicit = property(lambda self: 'user-defined' in self.namespaces,
                        doc=('Whether this namespace is explicit (i.e. '
                             'provided by the user at creation time and not '
                             'affected by the context in which it is run'))
    
    def add_namespace(self, name, namespace):
        try:
            namespace = dict(namespace)
        except TypeError:
            raise TypeError('namespace has to be mapping, is type %s' %
                            type(namespace))
        self.namespaces[name] = namespace
    
    def resolve(self, identifier, additional_namespace=None, strip_units=False):
        '''
        The additional_namespace (e.g. the local/global namespace) will only
        be used if the namespace does not contain any user-defined namespace.
        '''        
        # We save tuples of (namespace description, referred object) to
        # give meaningful warnings in case of duplicate definitions
        matches = []
        
        if self.is_explicit or additional_namespace is None: 
            namespaces = self.namespaces
        else:            
            namespaces = OrderedDict(self.namespaces)
            # Add the additional namespace in the end
            description, namespace = additional_namespace
            namespaces[description] = namespace
        
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
            if not all([(m[1] is first_obj) or _same_function(m[1], first_obj)
                        for m in matches]):
                _conflict_warning(('The name "%s" refers to different objects '
                                   'in different namespaces used for resolving. '
                                   'Will use the object from the %s namespace '
                                   'with the value %r') %
                                  (identifier, matches[0][0],
                                   first_obj), matches[1:])
                    
        # use the first match (according to resolution order)
        resolved = matches[0][1]

        # Remove units
        if strip_units and isinstance(resolved, Quantity):
            if resolved.ndim == 0:
                resolved = float(resolved)
            else:
                resolved = np.asarray(resolved)

        # Use standard Python types if possible
        if not isinstance(resolved, np.ndarray) and hasattr(resolved, 'dtype'):
            numpy_type = resolved.dtype
            if np.can_cast(numpy_type, np.int_):
                resolved = int(resolved)
            elif np.can_cast(numpy_type, np.float_):
                resolved = float(resolved)
            elif np.can_cast(numpy_type, np.complex_):
                resolved = complex(resolved)

        # Replace pure Python functions by a Functions object
        if callable(resolved) and not isinstance(resolved, Function):
            resolved = Function(resolved)

        return resolved

    def resolve_all(self, identifiers, additional_namespace=None,
                    strip_units=True):
        resolutions = {}
        for identifier in identifiers:            
            resolved = self.resolve(identifier, additional_namespace,
                                    strip_units=strip_units)            
            resolutions[identifier] = resolved                
        
        return resolutions

    def __getitem__(self, key):
        return self.resolve(key)
    
    def __setitem__(self, key, value):
        if not self.is_explicit:
            raise TypeError('This object does not have a user-defined '
                            'namespace, cannot add items directly.')
        self.namespaces['user-defined'][key] = value
    
    def __delitem__(self, key):
        if not self.is_explicit:
            raise TypeError('this object does not have a user-defined '
                            'namespace, cannot delete keys from it.')
        del self.namespaces['user-defined'][key]
    
    def __len__(self):
        total_length = 0
        for namespace in self.namespaces:
            total_length += len(self.namespaces[namespace])
        return total_length
    
    def __iter__(self):
        # do not repeat entries
        previous_entries = []
        for entries in self.namespaces.itervalues():
            for entry in entries:                
                if not entry in previous_entries:
                    previous_entries.append(entry)
                    yield entry
    
    def __contains__(self, key):
        for entries in self.namespaces.itervalues():
            if key in entries:
                return True
        
        return False
    
    def __repr__(self):
        return '<%s containing namespaces: %s>' % (self.__class__.__name__,
                                                   ', '.join(self.namespaces.iterkeys()))

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

_function_names = get_default_numpy_namespace().keys()
def check_identifier_functions(identifier):
    '''
    Make sure that identifier names do not clash with function names.
    '''
    if identifier in _function_names:
        raise ValueError('"%s" is the name of a function, cannot be used as a '
                         'variable name.')
        
equations.Equations.register_identifier_check(check_identifier_functions)

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

def check_identifier_units(identifier):
    '''
    Make sure that identifier names do not clash with unit names.
    '''
    if identifier in DEFAULT_UNIT_NAMESPACE:
        raise ValueError('"%s" is the name of a unit, cannot be used as a '
                         'variable name.')
        
equations.Equations.register_identifier_check(check_identifier_units)
