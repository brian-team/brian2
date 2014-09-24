'''
Runtime targets for code generation.
'''

from .numpy_rt import *
from .weave_rt import *
try:
    from .cython_rt import *
except ImportError:
    pass # todo: raise a warning?
