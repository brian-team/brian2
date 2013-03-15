Preferences system
==================

Each preference looks like ``codegen.c.compiler``, i.e. dotted names. Each
preference has to be registered and validated. The idea is that registering
all preferences ensures that misspellings of a preference value by a user
causes an error, e.g. if they wrote ``codgen.c.compiler`` it would raise an
error. Validation means that the value is checked for validity, so
``codegen.c.compiler = 'gcc'`` would be allowed, but
``codegen.c.compiler = 'hcc'`` would cause an error.

An additional requirement is that the preferences system allows for extension
modules to define their own preferences, including extending the existing
core brian preferences. For example, an extension might want to define
``extension.*`` but it might also want to define a new language for
codegen, e.g. ``codegen.lisp.*``.

Preference files
----------------

Preferences are stored in a hierarchy of files, with the following order
(each step overrides the values in the previous step but no error is raised
if one is missing):

* The global defaults are stored in the installation directory.
* The user default are stored in ``~/.brian/preferences`` (which works on
  Windows as well as Linux).
* The file ``brian_preferences`` in the current directory.

Registration
------------

Registration of preferences is performed by a call to `register_preferences`,
e.g.::

    register_preferences(
        'codegen.c',
        {'compiler': BrianPreference(
            validator=is_compiler,
            docs='...',
            default='gcc'),
         ...
        })
        
The first argument ``'codegen.c'`` is the base name, and every preference of
the form ``codegen.c.*`` has to be registered by this function (although you
can also register, e.g. ``codegen.c.somethingelse.*`` separately). In other
words, by calling `register_preferences`, a module takes ownership of all
the preferences with one particular base name.

The second argument is a dictionary of name/definition pairs in a fairly
obvious way. Alternatively, we could make it look like::

    register_preferences(
        'codegen.c',
        compiler=BrianPreference(
            validator=is_compiler,
            docs='...',
            default='gcc'),
        ...
        )

Validation functions
--------------------

A validation function takes a string as an argument, and returns a value (e.g.
int or string). If the string is invalid, it raises a
`PreferenceValidationError` exception.

Validation
----------

Setting the value of a preference with a registered base name instantly triggers
validation. For base names that are not yet registered, validation occurs when
the base name is registered. If, at the time that `Network.run` is called, there
are unregistered preferences set, a `PreferenceValidationError` is raised.

File format
-----------

The preference files are of the following form::

	a.b.c = 1
	# Comment line
	[a]
	b.d = 2
	[a.b]
	b.e = 3
	
This would set preferences ``a.b.c=1``, ``a.b.d=2`` and ``a.b.e=3``.
 
Built-in preferences
--------------------
Brian itself defines the following preferences:

.. document_brian_prefs::
   :nolinks: