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

import brian2.units.unitsafefunctions as unitsafe
from brian2.utils.logger import get_logger
from brian2.units.fundamentalunits import Quantity, all_registered_units
from brian2.units.stdunits import stdunits
from brian2.codegen.functions.numpyfunctions import (FunctionWrapper,
                                                     RandnFunction)

__all__ = ['ObjectWithNamespace',
           'get_default_numpy_namespace',
           'DEFAULT_UNIT_NAMESPACE']

logger = get_logger(__name__)

class ObjectWithNamespace(object):
    def __new__(cls, *args, **kwds):
        instance = super(ObjectWithNamespace, cls).__new__(cls, *args, **kwds)
        frame = inspect.stack()[1][0]
        instance._locals = dict(frame.f_locals)
        instance._globals = dict(frame.f_globals)
        return instance
    
    def create_namespace(self, N, explicit_namespace=None):                
        namespace = CompoundNamespace()
        
        # Explicitly given namespace overwrites all other namespaces
        if explicit_namespace is not None:
            namespace.add_namespace('user-defined', explicit_namespace)
        
        namespace.add_namespace('numpy', get_default_numpy_namespace(N))
        namespace.add_namespace('units', DEFAULT_UNIT_NAMESPACE)
        
        # only use the local/global namespace if no explicit one is given
        if explicit_namespace is None:
            namespace.add_namespace('local', self._locals)
            namespace.add_namespace('global', self._globals)         
        
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
    `Function` specifiers whose `Function.pyfunc` attribute matches as the
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
    
    def add_namespace(self, name, namespace):
        try:
            namespace = dict(namespace)
        except TypeError:
            raise TypeError('namespace has to be mapping, is type %s' %
                            type(namespace))
        self.namespaces[name] = namespace
    
    def resolve(self, identifier):
        # We save tuples of (namespace description, referred object) to
        # give meaningful warnings in case of duplicate definitions
        matches = []
        
        namespaces = self.namespaces
        
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
        return matches[0][1]

    def resolve_all(self, identifiers, strip_units=True):
        resolutions = {}
        for identifier in identifiers:            
            resolved = self.resolve(identifier)
            if strip_units and isinstance(resolved, Quantity):
                if resolved.ndim == 0:
                    resolved = float(resolved)
                else:
                    resolved = np.asarray(resolved)
            resolutions[identifier] = resolved                
        
        return resolutions

    def __getitem__(self, key):
        return self.resolve(key)
    
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
        for entries in self.namespace.itervalues():
            if key in entries:
                return True
        
        return False

def get_default_numpy_namespace(N):
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
    namespace = {'pi': np.pi, 'e': np.e, 'inf': np.inf}
    
    # numpy functions that have the same name in numpy and math.h
    namespace.update({'cos': FunctionWrapper(unitsafe.cos),
                      'sin': FunctionWrapper(unitsafe.sin),
                      'tan': FunctionWrapper(unitsafe.tan),
                      'cosh': FunctionWrapper(unitsafe.cosh),
                      'sinh': FunctionWrapper(unitsafe.sinh),
                      'tanh': FunctionWrapper(unitsafe.tanh),
                      'exp': FunctionWrapper(unitsafe.exp),
                      'log': FunctionWrapper(unitsafe.log),
                      'log10': FunctionWrapper(unitsafe.log10),
                      'sqrt': FunctionWrapper(np.sqrt),
                      'ceil': FunctionWrapper(np.ceil),
                      'floor': FunctionWrapper(np.floor)
                      })
    
    # numpy functions that have a different name in numpy and math.h
    namespace.update({'arccos': FunctionWrapper(unitsafe.arccos, cpp_name='acos'),
                      'arcsin': FunctionWrapper(unitsafe.arcsin, cpp_name='asin'),
                      'arctan': FunctionWrapper(unitsafe.arctan, cpp_name='atan'),
                      'power': FunctionWrapper(np.power, cpp_name='pow'),
                      'abs': FunctionWrapper(np.abs, py_name='abs', cpp_name='fabs'),
                      'mod': FunctionWrapper(np.mod, py_name='mod', cpp_name='fmod')})
    
    # numpy functions that need special treatment 
    namespace.update({'randn': RandnFunction(N),
                      '_randn': np.random.randn})
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
    namespace = dict([(u.name, u) for u in all_registered_units()])
    namespace.update(stdunits)
    return namespace

DEFAULT_UNIT_NAMESPACE = _get_default_unit_namespace()
