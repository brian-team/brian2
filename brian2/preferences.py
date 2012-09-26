'''
Brian global preferences are stored as attributes of a :class:`BrianGlobalPreferences`
object ``brian_prefs``.
'''

from .utils.stringtools import deindent, indent
from .units import have_same_dimensions, Quantity, DimensionMismatchError
import copy

__all__ = ['brian_prefs']

class DefaultValidator(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value
    def __call__(self, value):
        if not isinstance(value, self.value.__class__):
            try:
                valstr = str(value)
            except:
                valstr = '(undefined)'
            raise TypeError('Parameter {self.name} should be of class '
                            '{self.value.__class__.__name__}, value {valstr} '
                            'has class {value.__class__.__name__}'.format(
                                   self=self, value=value, valstr=valstr))
        if isinstance(self.value, Quantity):
            if not have_same_dimensions(self.value, value):
                raise DimensionMismatchError('Parameter {self.name}, current '
                                             'value = {self.value}, tried to '
                                             'change to {value}'.format(
                                                self=self, value=value))


class BrianGlobalPreferences(object):
    '''
    Class of the ``brian_prefs`` object, allows you to get, set and define
    global Brian preferences, and generate documentation.
    '''
    def __init__(self):
        self._docs = {}
        self._values = {}
        self._default_values = {}
        self._validators = {}
        self._backup_copy = {}
        self._initialised = True
    
    def _backup(self):
        '''
        Store a backup copy of the preferences to restore with `_restore`.
        '''
        self._backup_copy.update(
            docs=copy.copy(self._docs),
            values=copy.copy(self._values),
            defaults=copy.copy(self._default_values),
            validators=copy.copy(self._validators),
            )
    
    def _restore(self):
        '''
        Restore a copy of the values of the preferences backed up with `_backup`.
        '''
        self._docs.clear()
        self._values.clear()
        self._default_values.clear()
        self._validators.clear()
        self._docs.update(self._backup_copy['docs'])
        self._values.update(self._backup_copy['values'])
        self._default_values.update(self._backup_copy['defaults'])
        self._validators.update(self._backup_copy['validators'])
        
    def __getattr__(self, name):
        vals = super(BrianGlobalPreferences, self).__getattribute__('_values')
        return vals[name]
    
    def __setattr__(self, name, val):
        if not hasattr(self, '_initialised'):
            return super(BrianGlobalPreferences, self).__setattr__(name, val)
        if name in self._values:
            self._validators[name](val)
            self._values[name] = val
        else:
            raise KeyError("Preference %s is undefined, use define()"%name)
        
    def define(self, name, value, doc, validator=None):
        '''
        Define a new preference.
        
        ``name``
            The name of the preference
        ``value``
            The default value
        ``doc``
            A docstring describing the preference
        ``validator``
            An optional function ``validator(value)`` that checks that ``value``
            is an appropriate value for this preference, and raises an error
            if not. By default, it will check that the class of the value is
            derived from the class of the default value, and if it is a
            :class:`Quantity` it will check that the units match the default
            value.
        '''
        if name in self._values:
            raise KeyError("Preference %s already defined.")
        self._values[name] = value
        self._default_values[name] = value
        self._docs[name] = doc
        if validator is None:
            validator = DefaultValidator(name, value)
        self._validators[name] = validator

    def _get_documentation(self):
        s = ''
        for name in sorted(self._values.keys()):
            default = str(self._default_values[name])
            doc = str(self._docs[name])
            # Make a link target
            s += '.. _brian-pref-{name}:\n\n'.format(name=name.replace('_', '-'))
            s += '``{name}`` = ``{default}``\n'.format(name=name,
                                                       default=repr(default))
            s += indent(deindent(doc))
            s += '\n\n'
        return s
    
    documentation = property(fget=_get_documentation,
                             doc='Get a restructuredtext format documentation '
                                 'string for the defined parameters')

            
brian_prefs = BrianGlobalPreferences()
