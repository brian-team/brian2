'''
Module declaring general code generation preferences.
'''

from .codeobject import CodeObject
from brian2.core.preferences import brian_prefs, BrianPreference

# Preferences
brian_prefs.register_preferences(
    'codegen',
    'Code generation preferences',
    target = BrianPreference(
        default='numpy',
        docs='''
        Default target for code generation.
        
        Can be a string, in which case it should be one of:
        
        * `'numpy'` by default because this works on all platforms, but may not
          be maximally efficient.
        * `'weave`' uses ``scipy.weave`` to generate and compile C++ code,
          should work anywhere where ``gcc`` is installed and available at the
          command line.
        
        Or it can be a ``CodeObject`` class.
        ''',
        validator=lambda target: isinstance(target, str) or issubclass(target, CodeObject),
        ),
    )
