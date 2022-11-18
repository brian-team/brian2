"""
Module for transforming model equations into "abstract code" that can be then be
further translated into executable code by the `codegen` module.
"""
from .base import *
from .exact import *
from .explicit import *
from .exponential_euler import *
from .GSL import *

StateUpdateMethod.register("linear", linear)
StateUpdateMethod.register("exact", exact)
StateUpdateMethod.register("independent", independent)
StateUpdateMethod.register("exponential_euler", exponential_euler)
StateUpdateMethod.register("euler", euler)
StateUpdateMethod.register("rk2", rk2)
StateUpdateMethod.register("rk4", rk4)
StateUpdateMethod.register("milstein", milstein)
StateUpdateMethod.register("heun", heun)
StateUpdateMethod.register("gsl_rk2", gsl_rk2)
StateUpdateMethod.register("gsl_rk4", gsl_rk4)
StateUpdateMethod.register("gsl_rkf45", gsl_rkf45)
StateUpdateMethod.register("gsl_rkck", gsl_rkck)
StateUpdateMethod.register("gsl_rk8pd", gsl_rk8pd)
# as we consider rkf45 the default we also register it under 'gsl'
StateUpdateMethod.register("gsl", gsl_rkf45)

__all__ = [
    "StateUpdateMethod",
    "linear",
    "exact",
    "independent",
    "milstein",
    "heun",
    "euler",
    "rk2",
    "rk4",
    "ExplicitStateUpdater",
    "exponential_euler",
    "gsl_rk2",
    "gsl_rk4",
    "gsl_rkf45",
    "gsl_rkck",
    "gsl_rk8pd",
]
