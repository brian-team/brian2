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
from collections import MutableMapping

from brian2.utils.stringtools import deindent, indent
from brian2.units.fundamentalunits import (have_same_dimensions, Quantity,
                                           DimensionMismatchError)

__all__ = ['PreferenceError', 'BrianPreference', 'brian_prefs']

class PreferenceError(Exception):
    '''
    Exception relating to the Brian preferences system.
    '''
    pass


class DefaultValidator(object):
    '''
    Default preference validator
    
    Used by `BrianPreference` as the default validator if none is given.
    First checks if the provided value is of the same class as the default
    value, and then if the default is a `Quantity`, checks that the units
    match.
    '''
    def __init__(self, value):
        self.value = value
    def __call__(self, value):
        if not isinstance(value, self.value.__class__):
            return False
        if isinstance(self.value, Quantity):
            if not have_same_dimensions(self.value, value):
                return False
        return True


class BrianPreference(object):
    '''
    Used for defining a Brian preference.
    
    Parameters
    ----------
    
    default : object
        The default value.
    docs : str
        Documentation for the preference value.
    validator : func
        A function that True or False depending on whether the preference value
        is valid or not. If not specified, uses the `DefaultValidator` for the
        default value provided (check if the class is the same, and for
        `Quantity` objects, whether the units are consistent).
    representor : func
        A function that returns a string representation of a valid preference
        value that can be passed to `eval`. By default, uses `repr` which
        works in almost all cases.
    '''
    def __init__(self, default, docs, validator=None, representor=repr):
        self.representor = representor
        if validator is None:
            validator = DefaultValidator(default)
        self.validator = validator
        self.default = default
        self.docs = docs


class BrianGlobalPreferences(MutableMapping):
    '''
    Class of the ``brian_prefs`` object.
    
    Used for getting/setting/validating/registering preference values.
    All preferences must be registered via `register_preferences`. To get or
    set a preference, use the dictionary-like interface, e.g.::
    
        brian_prefs['core.default_scalar_dtype'] = float32
        
    Preferences can be read from files, see `load_preferences` and
    `read_preference_file`. Note that `load_preferences` is called
    automatically when Brian has finished importing.
    '''
    def __init__(self):
        self.prefs = {}
        self.backup_prefs = {}
        self.prefs_unvalidated = {}
        self.pref_register = {}
        self.eval_namespace = {}
        exec deindent('''
            from numpy import *
            from scipy import *
            from brian2.units import *
            from brian2.units.stdunits import *
            ''') in self.eval_namespace

    def __getitem__(self, item):
        return self.prefs[item]
    
    def __len__(self):
        return len(self.prefs)
    
    def __iter__(self):
        return iter(self.prefs)
    
    def __contains__(self, item):
        return item in self.prefs
    
    def __setitem__(self, name, value):
        basename, endname = self.parse_name(name)
        if basename not in self.pref_register:
            self.prefs_unvalidated[name] = value
            return
        prefdefs, _ = self.pref_register[basename]
        if endname in prefdefs:
            # do validation
            pref = prefdefs[endname]
            if not pref.validator(value):
                raise PreferenceError(
                    "Value %s for preference %s is invalid." % (value, name))
            self.prefs[name] = value
            if name in self.prefs_unvalidated:
                del self.prefs_unvalidated[name]
        else:
            raise PreferenceError("Preference "+name+" is unregistered. "
                                  "Spelling error?")
        
    def __delitem__(self, item):
        raise PreferenceError("Preferences cannot be deleted.")
    
    def parse_name(self, name):
        '''
        Split a preference name into a base and end name
        
        Returns
        -------
        
        basename : str
            The first part of the name up to the final ``.``.
        endname : str
            The last part of the name from the final ``.`` onwards.
        '''
        # parse the name
        parts = name.split('.')
        basename = '.'.join(parts[:-1])
        endname = parts[-1]
        return basename, endname
    
    def eval_pref(self, value):
        '''
        Evaluate a string preference in the units namespace
        '''
        return eval(value, self.eval_namespace)
    
    def _backup(self):
        '''
        Store a backup copy of the preferences to restore with `_restore`.
        '''
        self.backup_prefs.update(**self.prefs)
    
    def _restore(self):
        '''
        Restore a copy of the values of the preferences backed up with `_backup`.
        '''
        self.prefs.update(**self.backup_prefs)
        
    def _get_documentation(self):
        s = ''
        for basename, (prefdefs, basedoc) in self.pref_register.items():
            s += basename+'\n'
            s += '"'*len(basename)+'\n\n'
            s += deindent(basedoc, docstring=True).strip()+'\n\n'
            for name in sorted(prefdefs.keys()):
                pref = prefdefs[name]
                name = basename+'.'+name
                linkname = name.replace('_', '-').replace('.','-')
                # Make a link target
                s += '.. _brian-pref-{name}:\n\n'.format(name=linkname)
                s += '``{name}`` = ``{default}``\n'.format(name=name,
                                                           default=pref.representor(pref.default))
                s += indent(deindent(pref.docs, docstring=True))
                s += '\n\n'
        return s
    
    documentation = property(fget=_get_documentation,
                             doc='Get a restructuredtext format documentation '
                                 'string for the defined parameters')
    
    def _as_pref_file(self, valuefunc):
        '''
        Helper function used to generate the preference file for the default or current preference values.
        '''
        s = ''
        for basename, (prefdefs, basedoc) in self.pref_register.items():
            s += '#'+'-'*79+'\n'
            s += '\n'.join(['# '+line for line in deindent(basedoc, docstring=True).strip().split('\n')])+'\n'
            s += '#'+'-'*79+'\n\n'
            s += '['+basename+']\n\n'
            for name in sorted(prefdefs.keys()):
                pref = prefdefs[name]
                s += '\n'.join(['# '+line for line in deindent(pref.docs, docstring=True).strip().split('\n')])+'\n\n'
                s += name + ' = '+pref.representor(valuefunc(pref, basename+'.'+name))+'\n\n'
        return s
    
    def _get_defaults_as_file(self):
        return self._as_pref_file(lambda pref, fullname: pref.default)

    defaults_as_file = property(fget=_get_defaults_as_file,
                       doc='Get a Brian preference doc file format '
                           'string for the default preferences')

    def _get_as_file(self):
        return self._as_pref_file(lambda pref, fullname: self[fullname])

    as_file = property(fget=_get_as_file,
                       doc='Get a Brian preference doc file format '
                           'string for the current preferences')
            
    def read_preference_file(self, file):
        '''
        Reads a Brian preferences file
    
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
            
        `eval` is called on the values, so strings should be written as, e.g.
        ``'3'`` rather than ``3``. The eval is called with all unit names
        available.
        Within a section, the section name is prepended to the key. So in the above
        example, it would give the following unvalidated dictionary::
        
            {'a.b.c': 1,
             'a.b.d': 2,
             'a.b.e': 3,
             }
        
        Parameters
        ----------
        
        file : file, str
            The file object or filename of the preference file.
        '''
        if isinstance(file, str):
            filename = file
            file = open(file, 'r')
        else:
            filename = repr(file)
        lines = file.readlines()
        file.close()
        # remove empty lines
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]
        # Remove comments
        lines = [line for line in lines if not line.startswith('#')]
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
                self[keyname] = self.eval_pref(value)
                continue
            # Otherwise raise a parsing error
            raise PreferenceError("Parsing error in preference file "+filename)
        

    def load_preferences(self):
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
                self.read_preference_file(file)
            except IOError:
                pass

    def register_preferences(self, prefbasename, prefbasedoc, **prefs):
        '''
        Registers a set of preference names, docs and validation functions.
        
        Parameters
        ----------
        
        prefbasename : str
            The base name of the preference.
        prefbasedoc : str
            Documentation for this base name
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
        if prefbasename in self.pref_register:
            raise PreferenceError("Base name "+prefbasename+" already registered.")
        self.pref_register[prefbasename] = (prefs, prefbasedoc)
        for k, v in prefs.items():
            self.prefs_unvalidated[prefbasename+'.'+k] = v.default
        self.do_validation()

    def do_validation(self):
        '''
        Validates preferences that have not yet been validated.
        '''
        for name, value in self.prefs_unvalidated.items():
            self[name] = value
            
    def check_all_validated(self):
        '''
        Checks that all preferences that have been set have been validated.
        
        Logs a warning if not. Should be called by `Network.run` or other
        key Brian functions.
        '''
        if len(self.prefs_unvalidated):
            from brian2.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warn("The following preferences values have been set but "
                        "are not registered preferences:\n%s\nThis is usually "
                        "because of a spelling mistake or missing library "
                        "import." % ', '.join(self.prefs_unvalidated.keys()),
                        once=True)


brian_prefs = BrianGlobalPreferences()


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
            brian_prefs.read_preference_file('test_prefs')
            print brian_prefs.prefs_unvalidated.items()
        except Exception as e:
            print 'EXCEPTION!', e
        os.remove('test_prefs')
    if 1:
        dummy_validator = lambda x: True
        brian_prefs.register_preferences('useless', '''useless docs
                                           on multiple lines
                                           ''',
            a_string=BrianPreference('"default"', 'docs\noh docs',
                                     validator=dummy_validator),
            )
        open('brian_preferences', 'w').write('''
            useless.a_string = "blah"
            [meh]
            feh = 5
            ''')
        try:
            brian_prefs.load_preferences()
            print brian_prefs.prefs_unvalidated
        except Exception as e:
            print "EXCEPTION!", e
        os.remove('brian_preferences')
        print
        brian_prefs.do_validation()
        print
        brian_prefs.register_preferences('meh', 'meh docs',
            feh=BrianPreference('2', 'doc', validator=dummy_validator),
            )
        print
        print 'VALIDATED PREFERENCES:'
        for k, v in brian_prefs.items():
            print k, '=', repr(v)
        print
        print 'UNVALIDATED PREFERENCES:'
        for k, v in brian_prefs.prefs_unvalidated.items():
            print k, '=', repr(v)
        print
        print brian_prefs.documentation
        print brian_prefs.as_file
