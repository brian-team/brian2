Preferences system
==================

.. currentmodule:: brian2.core.preferences

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
codegen, e.g. ``codegen.lisp.*``. However, extensions cannot add preferences
to an existing category.

Accessing and setting preferences
---------------------------------
Preferences can be accessed and set either keyword-based or attribute-based.
To set/get the value for the preference example mentioned before, the following
are equivalent::

    prefs['codegen.c.compiler'] = 'gcc'
    prefs.codegen.c.compiler = 'gcc'

    if prefs['codegen.c.compiler'] == 'gcc':
        ...
    if prefs.codegen.c.compiler == 'gcc':
        ...

Using the attribute-based form can be particulary useful for interactive
work, e.g. in ipython, as it offers autocompletion and documentation.
In ipython, ``prefs.codegen.c?`` would display a docstring with all
the preferences available in the ``codegen.c`` category.

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

Registration of preferences is performed by a call to
`BrianGlobalPreferences.register_preferences`, e.g.::

    register_preferences(
        'codegen.c',
        'Code generation preferences for the C language',
        'compiler'= BrianPreference(
            validator=is_compiler,
            docs='...',
            default='gcc'),
         ...
        )

The first argument ``'codegen.c'`` is the base name, and every preference of
the form ``codegen.c.*`` has to be registered by this function (preferences in subcategories
such as ``codegen.c.somethingelse.*`` have to be specified separately). In other
words, by calling `~BrianGlobalPreferences.register_preferences`,
a module takes ownership of all the preferences with one particular base name. The second argument
is a descriptive text explaining what this category is about. The preferences themselves are
provided as keyword arguments, each set to a `BrianPreference` object.

Validation functions
--------------------

A validation function takes a value for the preference and returns ``True`` (if the value is a valid
value) or ``False``. If no validation function is specified, a default validator is used that
compares the value against the default value: Both should belong to the same class (e.g. int or
str) and, in the case of a `Quantity` have the same unit.

Validation
----------

Setting the value of a preference with a registered base name instantly triggers
validation. Trying to set an unregistered preference using keyword or attribute access raises an
error. The only exception from this rule is when the preferences are read from configuration files
(see below). Since this happens before the user has the chance to import extensions that potentially
define new preferences, this uses a special function (`_set_preference`). In this case,for base
names that are not yet registered, validation occurs when
the base name is registered. If, at the time that `Network.run` is called, there
are unregistered preferences set, a `PreferenceError` is raised.

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
