'''
Module for transforming model equations into "abstract code" that can be then be
further translated into executable code by the `codegen` module.
'''  
from .base import *
from .exact import *
from .explicit import *
from .exponential_euler import *

# Register the standard state updaters in the order in which they should be
# chosen
StateUpdateMethod.register('linear', linear)
StateUpdateMethod.register('independent', independent)
StateUpdateMethod.register('exponential_euler', exponential_euler)
StateUpdateMethod.register('euler', euler)
StateUpdateMethod.register('rk2', rk2)
StateUpdateMethod.register('rk4', rk4)
StateUpdateMethod.register('milstein', milstein)
StateUpdateMethod.register('heun', heun)


