import sympy as sp

from brian2.parsing.sympytools import str_to_sympy, sympy_to_str

from .base import (
    StateUpdateMethod,
    UnsupportedEquationsException,
    extract_method_options,
)

__all__ = ["exponential_euler"]


def get_conditionally_linear_system(eqs, variables=None):
    """
    Convert equations into a linear system using sympy.

    Parameters
    ----------
    eqs : `Equations`
        The model equations.

    Returns
    -------
    coefficients : dict of (sympy expression, sympy expression) tuples
        For every variable x, a tuple (M, B) containing the coefficients M and
        B (as sympy expressions) for M * x + B

    Raises
    ------
    ValueError
        If one of the equations cannot be converted into a M * x + B form.

    Examples
    --------
    >>> from brian2 import Equations
    >>> eqs = Equations('''
    ... dv/dt = (-v + w**2.0) / tau : 1
    ... dw/dt = -w / tau : 1
    ... ''')
    >>> system = get_conditionally_linear_system(eqs)
    >>> print(system['v'])
    (-1/tau, w**2.0/tau)
    >>> print(system['w'])
    (-1/tau, 0)

    """
    diff_eqs = eqs.get_substituted_expressions(variables)

    coefficients = {}

    for name, expr in diff_eqs:
        var = sp.Symbol(name, real=True)

        s_expr = str_to_sympy(expr.code, variables).expand()
        if s_expr.has(var):
            # Factor out the variable
            s_expr = sp.collect(s_expr, var, evaluate=False)

            if len(s_expr) > 2 or var not in s_expr:
                raise ValueError(
                    f"The expression '{expr}', defining the variable "
                    f"'{name}', could not be separated into linear "
                    "components."
                )
            coefficients[name] = (s_expr[var], s_expr.get(1, 0))
        else:
            coefficients[name] = (0, s_expr)

    return coefficients


class ExponentialEulerStateUpdater(StateUpdateMethod):
    """
    A state updater for conditionally linear equations, i.e. equations where
    each variable only depends linearly on itself (but possibly non-linearly
    on other variables). Typical Hodgkin-Huxley equations fall into this
    category, it is therefore the default integration method used in the
    GENESIS simulator, for example.
    """

    def __call__(self, equations, variables=None, method_options=None):
        extract_method_options(method_options, {})
        if equations.is_stochastic:
            raise UnsupportedEquationsException(
                "Cannot solve stochastic equations with this state updater."
            )

        # Try whether the equations are conditionally linear
        try:
            system = get_conditionally_linear_system(equations, variables)
        except ValueError:
            raise UnsupportedEquationsException(
                "Can only solve conditionally linear systems with this state updater."
            ) from None

        code = []
        for var, (A, B) in system.items():
            s_var = sp.Symbol(var)
            s_dt = sp.Symbol("dt")
            if A == 0:
                update_expression = s_var + s_dt * B
            elif B != 0:
                BA = B / A
                # Avoid calculating B/A twice
                BA_name = f"_BA_{var}"
                s_BA = sp.Symbol(BA_name)
                code += [f"{BA_name} = {sympy_to_str(BA)}"]
                update_expression = (s_var + s_BA) * sp.exp(A * s_dt) - s_BA
            else:
                update_expression = s_var * sp.exp(A * s_dt)

            # The actual update step
            code += [f"_{var} = {sympy_to_str(update_expression)}"]

        # Replace all the variables with their updated value
        for var in system:
            code += [f"{var} = _{var}"]

        return "\n".join(code)

    # Copy doc from parent class
    __call__.__doc__ = StateUpdateMethod.__call__.__doc__


exponential_euler = ExponentialEulerStateUpdater()
