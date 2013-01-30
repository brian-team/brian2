'''
Implementation of the namespace system, used to resolve the identifiers in
model equations of `NeuronGroup` and `Synapses`
'''
import inspect
import collections
import numpy as np

from brian2.utils.logger import get_logger
import brian2.units.unitsafefunctions as usf
from brian2.units.fundamentalunits import all_registered_units
from brian2.units.stdunits import stdunits
__all__ = ['Namespace', 'NamespaceView']

logger = get_logger(__name__)

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


class Namespace(collections.MutableMapping):
    '''
    refers : list of 3-tuples
        A list containing a tuples of the form (`name`, `suffix list`, `namespace`)
    '''
    
    def __init__(self, model_variables, namespace=None, exhaustive=False,
                 level=1, refers=None):        
        
        self._namespaces = collections.OrderedDict()
        
        self._namespaces['model'] = dict(model_variables)
        
        if namespace is None:
            # The user-defined namespace always exists, it can be extended later
            namespace = {}
        self._namespaces['user-defined'] = dict(namespace)
        
        if refers is not None:
            try:
                refers = tuple(refers)
            except TypeError:
                raise TypeError(('refers argument has to be a sequence but is '
                                'type %{type}').format(type=type(refers)))
                
            # only save views to the referred namespaces
            for refer_entry in refers:
                try:
                    name, suffixes, referred = refer_entry
                    self._namespaces[name] = NamespaceView(referred, suffixes)
                except ValueError:
                    raise TypeError('refers argument has to contain tuples '
                                    'with 3 entries.')
                
        
        if exhaustive:
            # do not use the implict namespace
            self.locals = {}
            self.globals = {}
        else:
            frame = inspect.stack()[level + 1][0]
            self._namespaces['locals'] = dict(frame.f_locals)
            self._namespaces['globals'] = dict(frame.f_globals)
        
        # add the standard namespaces for units and functions
        self._namespaces['units'] = DEFAULT_UNIT_NAMESPACE
        self._namespaces['numpy'] = DEFAULT_NUMPY_NAMESPACE
    
    def resolve(self, identifier):
        # We save tuples of (namespace description, referred object) to
        # give meaningful warnings in case of duplicate definitions
        matches = []
        
        namespaces = self._namespaces
        
        for description, namespace in namespaces.iteritems():
            if identifier in namespace:
                matches.append((description, namespace[identifier]))            

        if len(matches) == 0:
            # No match at all
            raise ValueError(('The identifier "%s" could not be resolved.') % 
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

    def resolve_all(self, identifiers):
        resolutions = {}
        for identifier in identifiers:
            resolutions[identifier] = self.resolve(identifier)
        return resolutions

    def __getitem__(self, key):
        return self.resolve(key)

    def __setitem__(self, key, value):
        # setting a value only affects the user-defined namespace
        self._namespaces['user-defined'][key] = value
    
    def __delitem__(self, key):
        if key in self._namespaces['user-defined']:
            del self._namespaces['user-defined'][key]
        else:
            raise KeyError('Unknown key "%s"' % key)
    
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

class NamespaceView(collections.Mapping):
    
    def __init__(self, namespace, suffixes=None):
        # TODO: We can't use weak references here, unfortunately
        self.namespace = namespace
        
        if suffixes is None:
            # make namespace lookup work without any suffixes
            suffixes = ['']

        self.suffixes = suffixes
    
    def __getitem__(self, key):
        for suffix in self.suffixes:
            if key.endswith(suffix):
                key_without_suffix = key[:key.rfind(suffix)]
                if (key_without_suffix) in self.namespace:
                    return self.namespace[key_without_suffix]
        
        raise KeyError('Illegal key %s' % key)
    
    def __len__(self):
        return len(self.namespace)
    
    def __contains__(self, key):
        for suffix in self.suffixes:
            if key.endswith(suffix):
                key_without_suffix = key[:key.rfind(suffix)]
                if (key_without_suffix) in self.namespace:
                    return True
        return False
    
    def __iter__(self):
        for suffix in self.suffixes:
            for key in self.namespace.iterkeys():
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

DEFAULT_NUMPY_NAMESPACE = NamespaceView(_get_default_numpy_namespace())


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

DEFAULT_UNIT_NAMESPACE = NamespaceView(_get_default_unit_namespace())
