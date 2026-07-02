"""
Runtime targets for code generation.
"""

# Register the base category before importing the individual codegen targets with
# their subcategories

from brian2.core.preferences import prefs
from brian2.utils.logger import get_logger

prefs.register_preferences(
    "codegen.runtime",
    "Runtime codegen preferences (see subcategories for individual targets)",
)

logger = get_logger(__name__)

# Always available
from .numpy_rt import *

# Optional: Cython (requires Cython + C++ compiler)
try:
    from .cython_rt import *
except ImportError:
    logger.debug("Cython runtime not available", exc_info=True)

# Optional: cppyy (requires cppyy, no external compiler needed)
try:
    from .cppyy_rt import *
except ImportError:
    logger.debug("cppyy runtime not available", exc_info=True)

# Optional: GSL integration
try:
    from .GSLcython_rt import *
except ImportError:
    pass
