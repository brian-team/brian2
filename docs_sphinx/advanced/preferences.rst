Preferences
===========

Brian has a system of global preferences that affect how certain objects
behave. These can be set either in scripts by using the `prefs` object
or in a file. Each preference looks like ``codegen.c.compiler``, i.e. dotted
names.

Accessing and setting preferences
---------------------------------
Preferences can be accessed and set either keyword-based or attribute-based.
The following are equivalent::

    prefs['codegen.c.compiler'] = 'gcc'
    prefs.codegen.c.compiler = 'gcc'

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
* The user default are stored in ``~/.brian/user_preferences`` (which works on
  Windows as well as Linux). The ``~`` symbol refers to the user directory.
* The file ``brian_preferences`` in the current directory.

The preference files are of the following form::

    a.b.c = 1
    # Comment line
    [a]
    b.d = 2
    [a.b]
    b.e = 3
    
This would set preferences ``a.b.c=1``, ``a.b.d=2`` and ``a.b.e=3``.
 
List of preferences
-------------------
Brian itself defines the following preferences (including their default
values):

.. document_brian_prefs::
