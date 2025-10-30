"""
Utility functions for parsing expressions and statements.
"""

import re
from collections import Counter

import sympy
from sympy.printing.precedence import precedence
from sympy.printing.str import StrPrinter

from brian2.core.functions import DEFAULT_CONSTANTS, DEFAULT_FUNCTIONS, Function
from brian2.parsing.rendering import SympyNodeRenderer
from brian2.utils.caching import cached


def check_expression_for_multiple_stateful_functions(expr, variables):
    identifiers = re.findall(r"\w+", expr)
    # Don't bother counting if we don't have any duplicates in the first place
    if len(identifiers) == len(set(identifiers)):
        return
    identifier_count = Counter(identifiers)
    for identifier, count in identifier_count.items():
        var = variables.get(identifier, None)
        if count > 1 and isinstance(var, Function) and not var.stateless:
            raise NotImplementedError(
                f"The expression '{expr}' contains "
                f"more than one call of {identifier}. This "
                "is currently not supported since "
                f"{identifier} is a stateful function and "
                "its multiple calls might be "
                "treated incorrectly (e.g."
                "'rand() - rand()' could be "
                " simplified to "
                "'0.0')."
            )


def str_to_sympy(expr, variables=None):
    """
    Parses a string into a sympy expression. There are two reasons for not
    using `sympify` directly: 1) sympify does a ``from sympy import *``,
    adding all functions to its namespace. This leads to issues when trying to
    use sympy function names as variable names. For example, both ``beta`` and
    ``factor`` -- quite reasonable names for variables -- are sympy functions,
    using them as variables would lead to a parsing error. 2) We want to use
    a common syntax across expressions and statements, e.g. we want to allow
    to use `and` (instead of `&`) and function names like `ceil` (instead of
    `ceiling`).

    Parameters
    ----------
    expr : str
        The string expression to parse.
    variables : dict, optional
        Dictionary mapping variable/function names in the expr to their
        respective `Variable`/`Function` objects.

    Returns
    -------
    s_expr
        A sympy expression

    Raises
    ------
    SyntaxError
        In case of any problems during parsing.
    """
    if variables is None:
        variables = {}
    check_expression_for_multiple_stateful_functions(expr, variables)

    # We do the actual transformation in a separate function that is cached
    # If we cached `str_to_sympy` itself, it would also use the contents of the
    # variables dictionary as the cache key, while it is only used for the check
    # above and does not affect the translation to sympy
    return _str_to_sympy(expr)


@cached
def _str_to_sympy(expr):
    try:
        s_expr = SympyNodeRenderer().render_expr(expr)
    except (TypeError, ValueError, NameError) as ex:
        raise SyntaxError(
            f"Error during evaluation of sympy expression '{expr}'"
        ) from ex

    return s_expr


class CustomSympyPrinter(StrPrinter):
    """
    Printer that overrides the printing of some basic sympy objects. E.g.
    print "a and b" instead of "And(a, b)".
    """

    def _print_And(self, expr):
        return " and ".join([f"({self.doprint(arg)})" for arg in expr.args])

    def _print_Or(self, expr):
        return " or ".join([f"({self.doprint(arg)})" for arg in expr.args])

    def _print_Not(self, expr):
        if len(expr.args) != 1:
            raise AssertionError(f'"Not" with {len(expr.args)} arguments?')
        return f"not ({self.doprint(expr.args[0])})"

    def _print_Relational(self, expr):
        return (
            f"{self.parenthesize(expr.lhs, precedence(expr))} "
            f"{self._relationals.get(expr.rel_op) or expr.rel_op} "
            f"{self.parenthesize(expr.rhs, precedence(expr))}"
        )

    def _print_Function(self, expr):
        # Special workaround for the int function
        if expr.func.__name__ == "int_":
            return f"int({self.stringify(expr.args, ', ')})"
        elif expr.func.__name__ == "Mod":
            return f"(({self.doprint(expr.args[0])})%({self.doprint(expr.args[1])}))"
        else:
            return f"{expr.func.__name__}({self.stringify(expr.args, ', ')})"


PRINTER = CustomSympyPrinter()


@cached
def sympy_to_str(sympy_expr):
    """
    sympy_to_str(sympy_expr)

    Converts a sympy expression into a string. This could be as easy as
    ``str(sympy_exp)`` but it is possible that the sympy expression contains
    functions like ``Abs`` (for example, if an expression such as
    ``sqrt(x**2)`` appeared somewhere). We do want to re-translate ``Abs`` into
    ``abs`` in this case.

    Parameters
    ----------
    sympy_expr : sympy.core.expr.Expr
        The expression that should be converted to a string.

    Returns
    str_expr : str
        A string representing the sympy expression.
    """
    # replace the standard functions by our names if necessary
    replacements = {
        f.sympy_func: sympy.Function(name)
        for name, f in DEFAULT_FUNCTIONS.items()
        if f.sympy_func is not None
        and isinstance(f.sympy_func, sympy.FunctionClass)
        and str(f.sympy_func) != name
    }
    # replace constants with our names as well
    replacements.update(
        {
            c.sympy_obj: sympy.Symbol(name)
            for name, c in DEFAULT_CONSTANTS.items()
            if str(c.sympy_obj) != name
        }
    )

    # Replace the placeholder argument by an empty symbol
    replacements[sympy.Symbol("_placeholder_arg")] = sympy.Symbol("")
    atoms = sympy_expr.atoms() | {f.func for f in sympy_expr.atoms(sympy.Function)}
    for old, new in replacements.items():
        if old in atoms:
            sympy_expr = sympy_expr.subs(old, new)
    expr = PRINTER.doprint(sympy_expr)

    return expr


def expression_complexity(expr, complexity=None):
    """
    Returns the complexity of an expression (either string or sympy)

    The complexity is defined as 1 for each arithmetic operation except divide which is 2,
    and all other operations are 20. This can be overridden using the complexity
    argument.

    Note: calling this on a statement rather than an expression is likely to lead to errors.

    Parameters
    ----------
    expr: `sympy.Expr` or str
        The expression.
    complexity: None or dict (optional)
        A dictionary mapping expression names to their complexity, to overwrite default behaviour.

    Returns
    -------
    complexity: int
        The complexity of the expression.
    """
    if isinstance(expr, str):
        # we do this because sympy.count_ops doesn't handle inequalities (TODO: handle sympy as well str)
        for op in ["<=", ">=", "==", "<", ">"]:
            expr = expr.replace(op, "+")
        # work around bug with rand() and randn() (TODO: improve this)
        expr = expr.replace("rand()", "rand(0)")
        expr = expr.replace("randn()", "randn(0)")
    subs = {"ADD": 1, "DIV": 2, "MUL": 1, "SUB": 1}
    if complexity is not None:
        subs.update(complexity)
    ops = sympy.count_ops(expr, visual=True)
    for atom in ops.atoms():
        if hasattr(atom, "name"):
            subs[atom.name] = 20  # unknown operations assumed to have a large cost
    return ops.evalf(subs=subs)
