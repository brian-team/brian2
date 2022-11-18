"""
Implementation of `BinomialFunction`
"""

import numpy as np

from brian2.core.base import Nameable
from brian2.core.functions import Function, DEFAULT_FUNCTIONS
from brian2.units.fundamentalunits import check_units
from brian2.utils.stringtools import replace

__all__ = ["BinomialFunction"]


def _pre_calc_constants_approximated(n, p):
    loc = n * p
    scale = np.sqrt(n * p * (1 - p))
    return loc, scale


def _pre_calc_constants(n, p):
    reverse = p > 0.5
    if reverse:
        P = 1.0 - p
    else:
        P = p
    q = 1.0 - P
    qn = np.exp(n * np.log(q))
    bound = min(n, n * P + 10.0 * np.sqrt(n * P * q + 1))
    return reverse, q, P, qn, bound


def _generate_cython_code(n, p, use_normal, name):
    # Cython implementation
    # Inversion transform sampling
    if use_normal:
        loc, scale = _pre_calc_constants_approximated(n, p)
        cython_code = """
        cdef float %NAME%(const int vectorisation_idx):
            return _randn(vectorisation_idx) * %SCALE% + %LOC%
        """
        cython_code = replace(
            cython_code,
            {"%SCALE%": f"{scale:.15f}", "%LOC%": f"{loc:.15f}", "%NAME%": name},
        )
        dependencies = {"_randn": DEFAULT_FUNCTIONS["randn"]}
    else:
        reverse, q, P, qn, bound = _pre_calc_constants(n, p)
        # The following code is an almost exact copy of numpy's
        # rk_binomial_inversion function
        # (numpy/random/mtrand/distributions.c)
        cython_code = """
        cdef long %NAME%(const int vectorisation_idx):
            cdef long X = 0
            cdef double px = %QN%
            cdef double U = _rand(vectorisation_idx)
            while U > px:
                X += 1
                if X > %BOUND%:
                    X = 0
                    px = %QN%
                    U = _rand(vectorisation_idx)
                else:
                    U -= px
                    px = ((%N%-X+1) * %P% * px)/(X*%Q%)
            return %RETURN_VALUE%
        """
        cython_code = replace(
            cython_code,
            {
                "%N%": f"{int(n)}",
                "%P%": f"{p:.15f}",
                "%Q%": f"{q:.15f}",
                "%QN%": f"{qn:.15f}",
                "%BOUND%": f"{bound:.15f}",
                "%RETURN_VALUE%": f"{int(n)}-X" if reverse else "X",
                "%NAME%": name,
            },
        )
        dependencies = {"_rand": DEFAULT_FUNCTIONS["rand"]}

    return cython_code, dependencies


def _generate_cpp_code(n, p, use_normal, name):
    # C++ implementation
    # Inversion transform sampling
    if use_normal:
        loc, scale = _pre_calc_constants_approximated(n, p)
        cpp_code = """
        float %NAME%(const int vectorisation_idx)
        {
            return _randn(vectorisation_idx) * %SCALE% + %LOC%;
        }
        """
        cpp_code = replace(
            cpp_code,
            {"%SCALE%": f"{scale:.15f}", "%LOC%": f"{loc:.15f}", "%NAME%": name},
        )
        dependencies = {"_randn": DEFAULT_FUNCTIONS["randn"]}
    else:
        reverse, q, P, qn, bound = _pre_calc_constants(n, p)
        # The following code is an almost exact copy of numpy's
        # rk_binomial_inversion function
        # (numpy/random/mtrand/distributions.c)
        cpp_code = """
        long %NAME%(const int vectorisation_idx)
        {
            long X = 0;
            double px = %QN%;
            double U = _rand(vectorisation_idx);
            while (U > px)
            {
                X++;
                if (X > %BOUND%)
                {
                    X = 0;
                    px = %QN%;
                    U = _rand(vectorisation_idx);
                } else
                {
                    U -= px;
                    px = ((%N%-X+1) * %P% * px)/(X*%Q%);
                }
            }
            return %RETURN_VALUE%;
        }
        """
        cpp_code = replace(
            cpp_code,
            {
                "%N%": f"{int(n)}",
                "%P%": f"{P:.15f}",
                "%Q%": f"{q:.15f}",
                "%QN%": f"{qn:.15f}",
                "%BOUND%": f"{bound:.15f}",
                "%RETURN_VALUE%": f"{int(n)}-X" if reverse else "X",
                "%NAME%": name,
            },
        )
        dependencies = {"_rand": DEFAULT_FUNCTIONS["rand"]}

    return {"support_code": cpp_code}, dependencies


class BinomialFunction(Function, Nameable):
    """
    BinomialFunction(n, p, approximate=True, name='_binomial*')

    A function that generates samples from a binomial distribution.

    Parameters
    ----------
    n : int
        Number of samples
    p : float
        Probablility
    approximate : bool, optional
        Whether to approximate the binomial with a normal distribution if
        :math:`n p > 5 \wedge n (1 - p) > 5`. Defaults to ``True``.
    """

    #: Container for implementing functions for different targets
    #: This container can be extended by other codegeneration targets/devices
    #: The key has to be the name of the target, the value a function
    #: that takes three parameters (n, p, use_normal) and returns a tuple of
    #: (code, dependencies)
    implementations = {"cpp": _generate_cpp_code, "cython": _generate_cython_code}

    @check_units(n=1, p=1)
    def __init__(self, n, p, approximate=True, name="_binomial*"):
        Nameable.__init__(self, name)

        # Python implementation
        use_normal = approximate and (n * p > 5) and n * (1 - p) > 5
        if use_normal:
            loc = n * p
            scale = np.sqrt(n * p * (1 - p))

            def sample_function(vectorisation_idx):
                try:
                    N = len(vectorisation_idx)
                except TypeError:
                    N = int(vectorisation_idx)
                return np.random.normal(loc, scale, size=N)

        else:

            def sample_function(vectorisation_idx):
                try:
                    N = len(vectorisation_idx)
                except TypeError:
                    N = int(vectorisation_idx)
                return np.random.binomial(n, p, size=N)

        Function.__init__(
            self,
            pyfunc=lambda: sample_function(1),
            arg_units=[],
            return_unit=1,
            stateless=False,
            auto_vectorise=True,
        )

        self.implementations.add_implementation("numpy", sample_function)

        for target, func in BinomialFunction.implementations.items():
            code, dependencies = func(n=n, p=p, use_normal=use_normal, name=self.name)
            self.implementations.add_implementation(
                target, code, dependencies=dependencies, name=self.name
            )
