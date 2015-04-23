'''
Runtime targets for code generation.
'''
# Register the base category before importing the indivial codegen targets with
# their subcategories
from brian2.core.preferences import prefs
prefs.register_preferences('codegen.runtime',
                           ('Runtime codegen preferences (see subcategories '
                            'for individual targets)'))

from .numpy_rt import *
from .weave_rt import *
try:
    from .cython_rt import *
except ImportError:
    pass # todo: raise a warning?
