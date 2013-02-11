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
import brian2.units.unitsafefunctions as usf
from brian2.units.fundamentalunits import Quantity, all_registered_units
from brian2.units.stdunits import stdunits

__all__ = ['ObjectWithNamespace',
           'DEFAULT_NUMPY_NAMESPACE',
           'DEFAULT_UNIT_NAMESPACE']

logger = get_logger(__name__)

class ObjectWithNamespace(object):
    def __new__(cls, *args, **kwds):
        instance = super(ObjectWithNamespace, cls).__new__(cls, *args, **kwds)
        frame = inspect.stack()[1][0]
        instance._locals = dict(frame.f_locals)
        instance._globals = dict(frame.f_globals)
        return instance
    
    def create_namespace(self, specifiers, explicit_namespace=None,
                         additional_namespaces=None):
        
        # This has to directly refer to the specifiers dictionary
        # and not to a copy so it takes any later changes (i.e. additions
        # for reset and threshold) into account 
        namespace = ModelNamespace(specifiers)        
        
        # only use the local/global namespace if no explicit one is given
        if explicit_namespace is None:
            namespace.add_namespace(Namespace('local', self._locals))
            namespace.add_namespace(Namespace('global', self._globals))
        else:            
            explicit_namespace = Namespace('user-defined', explicit_namespace)
            namespace.add_namespace(explicit_namespace)
        
        if not additional_namespaces is None:
            for additional in additional_namespaces:
                namespace.add_namespace(additional)
        
        namespace.add_namespace(DEFAULT_NUMPY_NAMESPACE)
        namespace.add_namespace(DEFAULT_UNIT_NAMESPACE)
        
        return namespace
    
    namespace = property(lambda self: self._namespace)

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


class ModelNamespace(collections.Mapping):

    def __init__(self, model_namespace):        
        
        self.namespaces = OrderedDict()        
        self.namespaces['model'] = model_namespace
        
        self._has_writeable = False
    
    def add_namespace(self, namespace):
        if not isinstance(namespace, Namespace):
            raise TypeError(('The namespace argument has to be of type "Namespace" '
                             'is type %s instead.') % str(type(namespace)))        
        self.namespaces[namespace.name] = namespace
    
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
            if not all([m[1] is first_obj for m in matches]):
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
        for namespace in self._namespaces:
            total_length += len(self._namespaces[namespace])
        return total_length
    
    def __iter__(self):
        # do not repeat entries
        previous_entries = []
        for entries in self._namespaces.itervalues():
            for entry in entries:                
                if not entry in previous_entries:
                    previous_entries.append(entry)
                    yield entry
    
    def __contains__(self, key):
        for entries in self._namespace.itervalues():
            if key in entries:
                return True
        
        return False


class Namespace(collections.Mapping):
    
    def __init__(self, name, namespace, suffixes=None):
        self.name = name
        if isinstance(namespace, ModelNamespace):
            self.namespace = namespace.namespaces['model']
        else:
            self.namespace = namespace
        self.suffixes = suffixes
    
    def __getitem__(self, key):
        if self.suffixes is None:
            return self.namespace[key]
        
        for suffix in self.suffixes:
            if key.endswith(suffix):
                key_without_suffix = key[:key.rfind(suffix)]
                if (key_without_suffix) in self.namespace:
                    return self.namespace[key_without_suffix]
        
        raise KeyError('Illegal key %s' % key)
    
    def __len__(self):
        return len(self.namespace)
    
    def __contains__(self, key):
        if self.suffixes is None:
            return (key in self.namespace)
        
        for suffix in self.suffixes:
            if key.endswith(suffix):
                key_without_suffix = key[:key.rfind(suffix)]
                if (key_without_suffix) in self.namespace:
                    return True
        return False
    
    def __iter__(self):
        if self.suffixes is None:
            for key in self.namespace:
                yield key
        else:
            for suffix in self.suffixes:
                for key in self.namespace:
                    yield key + suffix


def _get_default_numpy_namespace():
    '''
    Get the namespace of numpy functions/variables that is recognized by
    default. The namespace includes the constants :np:attr:`pi`,
    :np:attr:`e` and :np:attr:`inf` and the following functions:
    :np:func:`abs`, :np:func:`arccos`, :np:func:`arccosh`,
    :np:func:`arcsin`, :np:func:`arcsinh`, :np:func:`arctan`,
    :np:func:`arctanh`, :np:func:`ceil`, :np:func:`clip`,
    :np:func:`cos`, :np:func:`cosh`, :np:func:`exp`,
    :np:func:`floor`, :np:func:`log`, :np:func:`max`,
    :np:func:`mean`, :np:func:`min`, :np:func:`prod`,
    :np:func:`round`, :np:func:`sin`, :np:func:`sinh`,
    :np:func:`std`, :np:func:`sqrt`, :np:func:`sum`,
    :np:func:`tan`, :np:func:`tanh`, :np:func:`var`, :np:func:`where`
    
    Returns
    -------
    namespace : dict
        A dictionary mapping function/variable names to numpy objects or
        their unitsafe Brian counterparts.
    '''        
    # numpy constants
    namespace = {'pi': np.pi, 'e': np.e, 'inf': np.inf}
    
    # standard numpy functions
    numpy_funcs = [np.abs, np.floor, np.ceil, np.round, np.min, np.max,
                   np.mean, np.std, np.var, np.sum, np.prod, np.clip, np.sqrt]
    namespace.update([(func.__name__, func) for func in numpy_funcs])
    
    # unitsafe replacements for numpy functions
    replacements = [usf.log, usf.exp, usf.sin, usf.cos, usf.tan, usf.sinh,
                    usf.cosh, usf.tanh, usf.arcsin, usf.arccos, usf.arctan,
                    usf.arcsinh, usf.arccosh, usf.arctanh, usf.where]
    namespace.update([(func.__name__, func) for func in replacements])
    
    return namespace

DEFAULT_NUMPY_NAMESPACE = Namespace('numpy', _get_default_numpy_namespace())


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

DEFAULT_UNIT_NAMESPACE = Namespace('units',_get_default_unit_namespace())
