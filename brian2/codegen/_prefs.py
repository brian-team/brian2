'''
Module declaring general code generation preferences.

Preferences
-----------
.. document_brian_prefs:: codegen
'''

from .codeobject import CodeObject
from brian2.core.preferences import prefs, BrianPreference

# Preferences
prefs.register_preferences(
    'codegen',
    'Code generation preferences',
    target=BrianPreference(
        default='auto',
        docs='''
        Default target for code generation.
        
        Can be a string, in which case it should be one of:
        
        * ``'auto'`` the default, automatically chose the best code generation
          target available.
        * ``'weave'`` uses ``scipy.weave`` to generate and compile C++ code,
          should work anywhere where ``gcc`` is installed and available at the
          command line.
        * ``'cython'``, uses the Cython package to generate C++ code. Needs a
          working installation of Cython and a C++ compiler.
        * ``'numpy'`` works on all platforms and doesn't need a C compiler but
          is often less efficient.
        
        Or it can be a ``CodeObject`` class.
        ''',
        validator=lambda target: isinstance(target, basestring) or issubclass(target, CodeObject),
        ),
    string_expression_target=BrianPreference(
        default='numpy',
        docs='''
        Default target for the evaluation of string expressions (e.g. when
        indexing state variables). Should normally not be changed from the
        default numpy target, because the overhead of compiling code is not
        worth the speed gain for simple expressions.

        Accepts the same arguments as `codegen.target`, except for ``'auto'``
        ''',
        validator=lambda target: isinstance(target, basestring) or issubclass(target, CodeObject),
        ),
    loop_invariant_optimisations=BrianPreference(
        default=True,
        docs='''
        Whether to pull out scalar expressions out of the statements, so that
        they are only evaluated once instead of once for every neuron/synapse/...
        Can be switched off, e.g. because it complicates the code (and the same
        optimisation is already performed by the compiler) or because the
        code generation target does not deal well with it. Defaults to ``True``.
        '''
    ),
    max_cache_dir_size=BrianPreference(
      default=1000,
      docs='''
      The size of a directory (in MB) with cached code for weave or Cython that triggers a warning.
      Set to 0 to never get a warning.
      '''
    )
)