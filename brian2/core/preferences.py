'''
Brian global preferences are stored as attributes of a :class:`BrianGlobalPreferences`
object ``brian_prefs``.

Built-in preferences
--------------------

.. document_brian_prefs::
'''
import copy
import re
import os

from brian2.utils.stringtools import deindent, indent
from brian2.units.fundamentalunits import have_same_dimensions, Quantity, DimensionMismatchError

__all__ = ['PreferenceError', 'BrianPreference', 'register_preferences', 'brian_prefs']


class PreferenceError(Exception):
    '''
    Exception relating to the Brian preferences system.
    '''
    pass


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


class BrianGlobalPreferences(dict):
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
        
        Parameters
        ----------
        
        name : str
            The name of the preference
        value
            The default value
        doc : str
            A docstring describing the preference
        validator : function, optional
            An optional function ``validator(value)`` that checks that ``value``
            is an appropriate value for this preference, and raises an error
            if not. By default, it will check that the class of the value is
            derived from the class of the default value, and if it is a
            :class:`Quantity` it will check that the units match the default
            value.
        '''
        if name in self._values:            
            if (self._default_values[name] == value and self._docs[name] == doc
                and (self._validators[name] == validator or
                     (validator is None and
                      type(self._validators[name]) == DefaultValidator))):
                # This is a redefinition but exactly the same as before, just
                # ignore it.
                # This may happen during documentation generation where
                # __import__ is used, bypassing the standard import mechanism
                return
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
            default = self._default_values[name]
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

            
def read_preference_file(filename):
    '''
    Reads a Brian preferences file and returns a dictionary of key/value pairs

    The file format for Brian preferences is a plain text file of the form::

        a.b.c = 1
        # Comment line
        [a]
        b.d = 2
        [a.b]
        e = 3
        
    Blank and comment lines are ignored, all others should be of one of the
    following two forms::
    
        key = value
        [section]
        
    Within a section, the section name is prepended to the key. So in the above
    example, it would return the following dictionary::
    
        {'a.b.c': '1',
         'a.b.d': '2',
         'a.b.e': '3',
         }
    
    Parameters
    ----------
    
    filename : str
        The name of the preference file.
        
    Returns
    -------
    
    prefs : dict
        The extracted preferences, a dictionary of key/value pairs, where
        all of the values are unparsed strings.        
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    # remove empty lines
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    # Remove comments
    lines = [line for line in lines if not line.startswith('#')]
    prefs = {}
    bases = [] # start with no base
    for line in lines:
        # Match section names, which are used as a prefix for subsequent entries
        m = re.match('\[([^\]]*)\]', line)
        if m:
            bases = m.group(1).strip().split('.')
            continue
        # Match entries
        m = re.match('(.*?)=(.*)', line)
        if m:
            extname = m.group(1).strip()
            value = m.group(2).strip()
            keyname = '.'.join(bases+extname.split('.'))
            prefs[keyname] = value
            continue
        # Otherwise raise a parsing error
        raise ValueError("Parsing error in preference file "+file)
    return prefs


def load_preferences():
    '''
    Load all the preference files, but do not validate them.

    Preference files are read in the following order:
    
    1. ``brian2/default_preferences`` from the Brian installation directory.
    2. ``~/.brian/user_preferences`` from the user's home directory
    3. ``./brian_preferences`` from the current directory
    
    Files that are missing are ignored. Preferences read at each step
    override preferences from previous steps.
    
    See Also
    --------
    
    read_preference_file
    '''
    curdir, _ = os.path.split(__file__)
    basedir = os.path.normpath(os.path.join(curdir, '..'))
    default_prefs = os.path.join(basedir, 'default_preferences')
    user_prefs = os.path.join(os.path.expanduser('~'), '.brian/user_preferences')
    cur_prefs = 'brian_preferences'
    files = [default_prefs, user_prefs, cur_prefs]
    for file in files:
        try:
            newprefs = read_preference_file(file)
        except IOError:
            pass
        brian_prefs_unvalidated.update(**newprefs)


class BrianPreference(object):
    '''
    Used for defining a Brian preference.
    
    Parameters
    ----------
    
    default : str
        The default value as an unparsed string.
    docs : str
        Documentation for the preference value.
    validator : func
        A function that takes a string value as input, and returns the actual
        value (e.g. take ``'1'`` and return ``1``), or raises a
        `PreferenceError` if the string is invalid. TODO: default validators?
    '''
    def __init__(self, default, docs, validator=None):
        self.validator = validator
        self.default = default
        self.docs = docs


def register_preferences(prefbasename, **prefs):
    '''
    Registers a set of preference names, docs and validation functions.
    
    Parameters
    ----------
    
    prefbasename : str
        The base name of the preference.
    **prefs : dict of (name, `BrianPreference`) pairs
        The preference names to be defined. The full preference name will be
        ``prefbasename.name``, and the `BrianPreference` value is used to define
        the default value, docs, and validation function.
        
    Raises
    ------
    
    PreferenceError
        If the base name is already registered.
        
    See Also
    --------
    
    BrianPreference
    '''
    if prefbasename in pref_register:
        raise PreferenceError("Base name "+prefbasename+" already registered.")
    pref_register[prefbasename] = prefs
    do_validation()


def do_validation():
    '''
    Validates preferences that have not yet been validated.
    '''
    for name, valuestr in brian_prefs_unvalidated.items():
        # parse the name
        parts = name.split('.')
        basename = '.'.join(parts[:-1])
        endname = parts[-1]
        # if basename is not in pref_register, we ignore the preference for the
        # moment, do_validation() will get called by register_preferences
        if basename in pref_register:
            prefdefs = pref_register[basename]
            if endname in prefdefs:
                # do validation
                pref = prefdefs[endname]
                try:
                    value = pref.validator(valuestr)
                except Exception as e:
                    raise PreferenceError(
                          "Value '%s' for preference %s is invalid, "
                          "exception: %s"%(valuestr, name, str(e)))
                brian_prefs[name] = value
                del brian_prefs_unvalidated[name]
            else:
                raise PreferenceError("Preference "+name+" has "
                                      "an unregistered base name. Spelling "
                                      "error or forgotten library import?")


pref_register = {}
brian_prefs = BrianGlobalPreferences()
brian_prefs_unvalidated = {}


if __name__=='__main__':
    if 0:
        open('test_prefs', 'w').write('''
            a.b.c = 1
            # Comment line
            [a]
            b.d = 2
            [a.b]
            e = 3
            ''')
        try:
            prefs = read_preference_file('test_prefs')
            print prefs
        except Exception as e:
            print e
        os.remove('test_prefs')
    if 1:
        register_preferences('useless',
            a_string=BrianPreference('default', 'docs', validator=str),
            )
        open('brian_preferences', 'w').write('''
            useless.a_string = blah
            [meh]
            feh = 5
            ''')
        try:
            prefs = load_preferences()
            print brian_prefs_unvalidated
        except Exception as e:
            print e
        os.remove('brian_preferences')
        print
        do_validation()
        print
        register_preferences('meh',
            feh=BrianPreference('default', 'doc', validator=int),
            )
        print
        print 'VALIDATED PREFERENCES:'
        for k, v in brian_prefs.items():
            print k, '=', repr(v)
        print
        print 'UNVALIDATED PREFERENCES:'
        for k, v in brian_prefs_unvalidated.items():
            print k, '=', repr(v)
