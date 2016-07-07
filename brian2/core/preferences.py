'''
Brian global preferences are stored as attributes of a `BrianGlobalPreferences`
object ``prefs``.
'''
import itertools
import re
import os
from collections import MutableMapping
from StringIO import StringIO

from brian2.utils.stringtools import deindent, indent
from brian2.units.fundamentalunits import have_same_dimensions, Quantity

__all__ = ['PreferenceError', 'BrianPreference', 'prefs', 'brian_prefs']

def parse_preference_name(name):
    '''
    Split a preference name into a base and end name.
    
    Parameters
    ----------
    name : str
        The full name of the preference.
    
    Returns
    -------        
    basename : str
        The first part of the name up to the final ``.``.
    endname : str
        The last part of the name from the final ``.`` onwards.
    
    Examples
    --------
    >>> parse_preference_name('core.weave_compiler')
    ('core', 'weave_compiler')
    >>> parse_preference_name('codegen.cpp.compiler')
    ('codegen.cpp', 'compiler')
    '''
    # parse the name
    parts = name.split('.')
    basename = '.'.join(parts[:-1])
    endname = parts[-1]
    return basename, endname


def check_preference_name(name):
    '''
    Make sure that a preference name is valid. This currently checks that the
    name does not contain illegal characters and does not clash with method
    names such as "keys" or "items".
    
    Parameters
    ----------
    name : str
        The name to check.
    
    Raises
    ------
    PreferenceError
        In case the name is invalid.
    '''
    if not re.match("[A-Za-z][_a-zA-Z0-9]*$", name):
        raise PreferenceError(('Illegal preference name "%s": A preference '
                               'name can only start with a letter and only '
                               'contain letters, digits or underscore.' % name))
    if name in dir(MutableMapping) or name in prefs.__dict__:
        raise PreferenceError(('Illegal preference name "%s": This is also the '
                              'name of a method.') % name)


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
    Class of the ``prefs`` object.
    
    Used for getting/setting/validating/registering preference values.
    All preferences must be registered via `register_preferences`. To get or
    set a preference, you can either use a dictionary-based or an
    attribute-based interface::
    
        prefs['core.default_float_dtype'] = float32
        prefs.core.default_float_dtype = float32
        
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
            from brian2.units import *            
            from brian2.units.stdunits import *
            ''') in self.eval_namespace

    def __getitem__(self, item):
        if item in self.pref_register:
            # This asks for a category, not a single preference
            return BrianGlobalPreferencesView(item, self)
        return self.prefs[item]

    def __len__(self):
        return len(self.prefs)

    def __iter__(self):
        return iter(self.prefs)

    def __contains__(self, item):
        return item in self.prefs

    def __setitem__(self, name, value):
        basename, endname = parse_preference_name(name)
        if basename not in self.pref_register:
            raise PreferenceError("Preference category " + basename +
                                  " is unregistered. Spelling error?")
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
            raise PreferenceError("Preference " + name + " is unregistered. "
                                  "Spelling error?")

    def __delitem__(self, item):
        raise PreferenceError("Preferences cannot be deleted.")

    def __getattr__(self, name):
        if name in self.__dict__ or name.startswith('__'):
            return MutableMapping.__getattr__(self, name)
        
        # This function might get called from BrianGlobalPreferencesView with
        # a prefixed name -- therefore the name can contain dots!
        if name in self.pref_register:
            # This asks for a category, not a single preference
            return BrianGlobalPreferencesView(name, self)
        
        basename, _ = parse_preference_name(name)
        if len(basename) and basename not in self.pref_register:
            raise AssertionError(('__getattr__ received basename %s which is '
                                 'unregistered. This should never happen!') %
                                 basename)
        
        return self[name]

    def __setattr__(self, name, value):  
        # Do not allow to set a category name to something else
        if 'pref_register' in self.__dict__ and name in self.pref_register:
            raise PreferenceError('Cannot set a preference category.')
        else:
            MutableMapping.__setattr__(self, name, value)      

    def __delattr__(self, name):
        if 'pref_register' in self.__dict__ and name in self.pref_register:
            raise PreferenceError('Cannot delete a preference category.')
        else:
            MutableMapping.__setattr__(self, name, value)

    toplevel_categories = property(fget=lambda self: [category for category in
                                                      self.pref_register
                                                      if not '.' in category],
                                   doc='The toplevel preference categories')    

    def _get_docstring(self):
        '''
        Document the toplevel categories, used as a docstring for the object.
        '''
        s =  'Preference categories:\n\n'
        for category in self.toplevel_categories:
            s += '** %s **\n' % category
            _, category_doc = self.pref_register[category]
            s += '    ' + category_doc + '\n\n'
        
        return s

    def __dir__(self):
        res = dir(type(self)) + self.__dict__.keys()
        categories = self.toplevel_categories
        res.extend(categories)
        return res        

    def eval_pref(self, value):
        '''
        Evaluate a string preference in the units namespace
        '''
        return eval(value, self.eval_namespace)

    def _set_preference(self, name, value):
        '''
        Try to set the preference and allow for unregistered base names. This
        method is used internally when reading preferences from the file
        because the preferences are potentially defined in packages that are
        not imported yet. Unvalidated preferences are safed and will be
        validated as soon as the category is registered. `Network.run` will
        also check for unvalidated preferences.
        '''
        basename, _ = parse_preference_name(name)
        if basename not in self.pref_register:
            self.prefs_unvalidated[name] = value
        else:
            # go via the standard __setitem__ method
            self[name] = value
        

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

    def _get_one_documentation(self, basename, link_targets):
        '''
        Document a single category of preferences.
        '''

        s = ''
        if not basename in self.pref_register:
            raise ValueError('No preferences under the name "%s" are registered' % basename)
        prefdefs, basedoc = self.pref_register[basename]
        s += deindent(basedoc, docstring=True).strip() + '\n\n'
        for name in sorted(prefdefs.keys()):
            pref = prefdefs[name]
            name = basename + '.' + name
            linkname = name.replace('_', '-').replace('.', '-')
            if link_targets:
                # Make a link target
                s += '.. _brian-pref-{name}:\n\n'.format(name=linkname)
            s += '``{name}`` = ``{default}``\n'.format(name=name,
                                                       default=pref.representor(pref.default))
            s += indent(deindent(pref.docs, docstring=True))
            s += '\n\n'
        return s

    def get_documentation(self, basename=None, link_targets=True):
        '''
        Generates a string documenting all preferences with the given
        `basename`. If no `basename` is given, all preferences are documented.
        '''
        s = ''
        if basename is None:
            basenames = [tuple(basename.split('.')) for basename in self.pref_register.keys()]
            basenames.sort()
            for basename in basenames:
                lev = len(basename)
                basename = '.'.join(basename)
                if lev==1:
                    s += basename+'\n'+'"'*len(basename)+'\n\n'
                else:
                    s += '**' + basename + '**\n\n'
                s += self._get_one_documentation(basename, link_targets)
            #for basename in self.pref_register:
                #s += '**' + basename + '**\n\n'
                #s += basename+'\n'+'"'*len(basename)+'\n\n'
                #s += self._get_one_documentation(basename, link_targets)
        else:
            s += self._get_one_documentation(basename, link_targets)

        return s

    def _as_pref_file(self, valuefunc):
        '''
        Helper function used to generate the preference file for the default or current preference values.
        '''
        s = ''
        for basename, (prefdefs, basedoc) in self.pref_register.items():
            s += '#' + '-' * 79 + '\n'
            s += '\n'.join(['# ' + line for line in deindent(basedoc, docstring=True).strip().split('\n')]) + '\n'
            s += '#' + '-' * 79 + '\n\n'
            s += '[' + basename + ']\n\n'
            for name in sorted(prefdefs.keys()):
                pref = prefdefs[name]
                s += '\n'.join(['# ' + line for line in deindent(pref.docs, docstring=True).strip().split('\n')]) + '\n\n'
                s += name + ' = ' + pref.representor(valuefunc(pref, basename + '.' + name)) + '\n\n'
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
        bases = []  # start with no base
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
                keyname = '.'.join(bases + extname.split('.'))
                self._set_preference(keyname, self.eval_pref(value))
                continue
            # Otherwise raise a parsing error
            raise PreferenceError("Parsing error in preference file " + filename)

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
        user_prefs = os.path.join(os.path.expanduser('~'),
                                  '.brian/user_preferences')
        cur_prefs = 'brian_preferences'
        files = [default_prefs, user_prefs, cur_prefs]
        for file in files:
            try:
                self.read_preference_file(file)
            except IOError:
                pass
            
    def reset_to_defaults(self):
        '''
        Resets the parameters to their default values.
        '''
        self.read_preference_file(StringIO(self.defaults_as_file))

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
            # During the initial import phase the same base category may be
            # created twice, ignore that
            previous = self.pref_register[prefbasename]
            if not (len(previous[0]) == 0 and previous[1] == prefbasedoc):
                raise PreferenceError("Base name " + prefbasename +
                                      " already registered.")
        # Check that the new category does not clash with a preference name of
        # the parent category. For example, if a category "a" with the
        # preference "b" is already registered, do not allow to register a
        # preference category "a.b"
        basename, category_name = parse_preference_name(prefbasename)
        if len(basename) and basename in self.pref_register:
            parent_preferences, _ = self.pref_register[basename]
            if category_name in parent_preferences:
                raise PreferenceError(('Cannot register category "%s", '
                                       'parent category "%s" already has a '
                                       'preference named "%s".') %
                                      (prefbasename, basename, category_name))

        self.pref_register[prefbasename] = (prefs, prefbasedoc)
        for k, v in prefs.items():
            fullname = prefbasename + '.' + k
            # The converse of the above check: Check that a preference name
            # does not clash with a category 
            if fullname in self.pref_register:
                raise PreferenceError(('Cannot register "%s" as a preference, '
                                       'it is already registered as a '
                                       'preference category.') % fullname)
            check_preference_name(k)
            self.prefs_unvalidated[fullname] = v.default
        self.do_validation()
        
        # Update the docstring (a new toplevel category might have been added)
        self.__doc__ = self._get_docstring()

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

    def __repr__(self):
        description = '<{classname} with top-level categories: {categories}>'
        categories = ', '.join(['"%s"' % category for category
                                in self.toplevel_categories])
        return description.format(classname=self.__class__.__name__,
                                  categories=categories)


class BrianGlobalPreferencesView(MutableMapping):
    '''
    A class allowing for accessing preferences in a subcategory. It forwards
    requests to `BrianGlobalPreferences` and provides documentation and
    autocompletion support for all preferences in the given category. This
    object is used to allow accessing preferences via attributes of the
    `prefs` object.
    
    Parameters
    ----------
    basename : str
        The name of the preference category. Has to correspond to a key in
        `BrianGlobalPreferences.pref_register`.
    all_prefs : `BrianGlobalPreferences`
        A reference to the main object storing the preferences.
    '''
    
    def __init__(self, basename, all_prefs):
        self._basename = basename
        self._all_prefs = all_prefs
        self._subcategories = [key for key in all_prefs.pref_register.iterkeys()
                              if key.startswith(basename + '.')]
        self._preferences = all_prefs.pref_register[basename][0].keys()
        self.__doc__ = all_prefs.get_documentation(basename=basename,
                                                   link_targets=False)

    _sub_preferences = property(lambda self: [pref[len(self._basename+'.'):] for pref in self._all_prefs
                                              if pref.startswith(self._basename+'.')],
                                doc='All preferences in this category and its subcategories')

    def __getitem__(self, item):
        return self._all_prefs[self._basename + '.' + item]

    def __setitem__(self, item, value):
        self._all_prefs[self._basename + '.' + item] = value

    def __delitem__(self, item):
        raise PreferenceError("Preferences cannot be deleted.")
    
    def __len__(self):
        return len(self._sub_preferences)
    
    def __iter__(self):
        return iter(self._sub_preferences)

    def __contains__(self, item):
        return item in self._sub_preferences

    def __getattr__(self, name):
        return getattr(self._all_prefs, self._basename + '.' + name)

    def __setattr__(self, name, value):
        # Names starting with an underscore are not preferences but normal
        # instance attributes
        if name.startswith('_'):
            MutableMapping.__setattr__(self, name, value)
        else:
            self._all_prefs[self._basename + '.' + name] = value

    def __delattr__(self, name):
        # Names starting with an underscore are not preferences but normal
        # instance attributes
        if name.startswith('_'):
            MutableMapping.__delattr__(self, name)
        else:
            del self._all_prefs[self._basename + '.' + name]

    def __dir__(self):
        res = dir(type(self)) + self.__dict__.keys()
        res.extend(self._preferences)
        res.extend([category[len(self._basename+'.'):]
                    for category in self._subcategories])
        return res
    
    def __repr__(self):
        description = '<{classname} for preference category "{category}">'
        return description.format(classname=self.__class__.__name__,
                                  category=self._basename)

# : Object storing Brian's preferences
prefs = BrianGlobalPreferences()


# Simple class to give a useful error message when using `brian_prefs`
class ErrorRaiser(object):
    def __getattr__(self, item):
        raise AttributeError(("The global preferences object has been renamed "
                              "from 'brian_prefs' to 'prefs'"))

    def __getitem__(self, item):
        raise AttributeError(("The global preferences object has been renamed "
                              "from 'brian_prefs' to 'prefs'"))

brian_prefs = ErrorRaiser()
