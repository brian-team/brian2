'''
Runtime targets for code generation.
'''
import os

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


def _get_size_recursively(dirname):
    total_size = 0
    for dirpath, _, filenames in os.walk(dirname):
        for fname in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, fname))
    return total_size

def check_cache(target, cache_dir):
    size = _get_size_recursively(cache_dir)
    size_in_mb = int(round(size/1024./1024.))
    if size_in_mb > prefs.codegen.max_cache_dir_size:
        logger.warn('Cache size for target "%s": %s MB' % (target, size_in_mb))
    else:
        logger.debug('Cache size for target "%s": %s MB' % (target, size_in_mb))
