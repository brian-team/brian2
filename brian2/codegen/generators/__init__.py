# Register the base category before importing the indivial generators with
# their subcategories
from brian2.core.preferences import prefs
prefs.register_preferences('codegen.generators',
                           ('Codegen generator preferences (see subcategories '
                            'for individual languages)'))

from .base import *
from .cpp_generator import *
from .numpy_generator import *
try:
    from .cython_generator import *
except ImportError:
    pass # todo: raise a warning?

