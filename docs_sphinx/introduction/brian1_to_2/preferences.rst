Preferences (Brian 1 --> 2 conversion)
========================================
.. sidebar:: Brian 2 documentation

    For the main documentation about preferences, see the document
    :doc:`../../advanced/preferences`.

In Brian 1, preferences were set either with the function ``set_global_preferences`` or by creating a module
somewhere on the Python path called ``brian_global_config.py``.


Setting preferences
-------------------

The function ``set_global_preferences`` no longer exists in Brian 2. Instead, importing from ``brian2`` gives you a
variable `prefs` that can be used to set preferences. For example, in Brian 1 you would write::

    set_global_preferences(weavecompiler='gcc')

In Brian 2 you would write::

    prefs.codegen.cpp.compiler = 'gcc'

Configuration file
------------------

The module ``brian_global_config.py`` is not used by Brian 2, instead we search for configuration files in the
current directory, user directory or installation directory. In Brian you would have a configuration file that looks
like this::

    from brian.globalprefs import *
    set_global_preferences(weavecompiler='gcc')

In Brian 2 you would have a file like this::

    codegen.cpp.compiler = 'gcc'

Preference name changes
-----------------------

* ``defaultclock``: removed because it led to unclear behaviour of scripts.
* ``useweave_linear_diffeq``: removed because it was no longer relevant.
* ``useweave``: now replaced by `codegen.target` (but note that weave is no longer
  supported in Brian 2, use Cython instead).
* ``weavecompiler``: now replaced by `codegen.cpp.compiler`.
* ``gcc_options``: now replaced by `codegen.cpp.extra_compile_args_gcc`.
* ``openmp``: now replaced by `devices.cpp_standalone.openmp_threads`.
* ``usecodegen*``: removed because it was no longer relevant.
* ``usenewpropagate``: removed because it was no longer relevant.
* ``usecstdp``: removed because it was no longer relevant.
* ``brianhears_usegpu``: removed because Brian Hears doesn't exist in Brian 2.
* ``magic_useframes``: removed because it was no longer relevant.
