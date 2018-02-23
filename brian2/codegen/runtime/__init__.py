'''
Runtime targets for code generation.
'''

# Register the base category before importing the indivial codegen targets with
# their subcategories
from brian2.core.preferences import prefs
from brian2.utils.logger import get_logger
prefs.register_preferences('codegen.runtime',
                           ('Runtime codegen preferences (see subcategories '
                            'for individual targets)'))

logger = get_logger(__name__)

from .numpy_rt import *
from .weave_rt import *
from .GSLweave_rt import *
try:
    from .cython_rt import *
except ImportError:
    pass # todo: raise a warning?
try:
    from .GSLcython_rt import *
except ImportError:
    pass
