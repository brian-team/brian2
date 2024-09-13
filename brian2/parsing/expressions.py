"""
AST parsing based analysis of expressions
"""

import ast

from brian2.core.functions import Function
from brian2.parsing.rendering import NodeRenderer
from brian2.units.fundamentalunits import (
    DIMENSIONLESS,
    DimensionMismatchError,
    Unit,
    get_dimensions,
    get_unit_for_display,
    have_same_dimensions,
)

__all__ = ["parse_expression_dimensions"]


def is_boolean_expression(expr, variables):
    """
    Determines if an expression is of boolean type or not

    Parameters
    ----------

    expr : str
        The expression to test
    variables : dict-like of `Variable`
        The variables used in the expression.

    Returns
    -------
    isbool : bool
        Whether or not the expression is boolean.

    Raises
    ------
    SyntaxError
        If the expression ought to be boolean but is not,
        for example ``x<y and z`` where ``z`` is not a boolean variable.

    Notes
    -----
    We test the following cases recursively on the abstract syntax tree:

    * The node is a boolean operation. If all the subnodes are boolean
      expressions we return ``True``, otherwise we raise the ``SyntaxError``.
    * The node is a function call, we return ``True`` or ``False`` depending
      on whether the function description has the ``_returns_bool`` attribute
      set.
    * The node is a variable name, we return ``True`` or ``False`` depending
      on whether ``is_boolean`` attribute is set or if the name is ``True`` or
      ``False``.
    * The node is a comparison, we return ``True``.
    * The node is a unary operation, we return ``True`` if the operation is
      ``not``, otherwise ``False``.
    * Otherwise we return ``False``.
    """

    # If we are working on a string, convert to the top level node
    if isinstance(expr, str):
        mod = ast.parse(expr, mode="eval")
        expr = mod.body

    if expr.__class__ is ast.BoolOp:
        if all(is_boolean_expression(node, variables) for node in expr.values):
            return True
        else:
            raise SyntaxError(
                "Expression ought to be boolean but is not (e.g. 'x<y and 3')"
            )
    elif expr.__class__ is ast.Constant:
        value = expr.value
        if value is True or value is False:
            return True
        elif value is None:
            raise ValueError("Do not know how to deal with 'None'")
    elif expr.__class__ is ast.Name:
        name = expr.id
        if name in variables:
            return variables[name].is_boolean
        else:
            return name == "True" or name == "False"
    elif expr.__class__ is ast.Call:
        name = expr.func.id
        if name in variables and hasattr(variables[name], "_returns_bool"):
            return variables[name]._returns_bool
        else:
            raise SyntaxError(f"Unknown function {name}")
    elif expr.__class__ is ast.Compare:
        return True
    elif expr.__class__ is ast.UnaryOp:
        return expr.op.__class__.__name__ == "Not"
    else:
        return False


def _get_value_from_expression(expr, variables):
    """
    Returns the scalar value of an expression, and checks its validity.

    Parameters
    ----------
    expr : str or `ast.Expression`
        The expression to check.
    variables : dict of `Variable` objects
        The information about all variables used in `expr` (including `Constant`
        objects for external variables)

    Returns
    -------
    value : float
        The value of the expression

    Raises
    ------
    SyntaxError
        If the expression cannot be evaluated to a scalar value
    DimensionMismatchError
        If any part of the expression is dimensionally inconsistent.
    """
    # If we are working on a string, convert to the top level node
    if isinstance(expr, str):
        mod = ast.parse(expr, mode="eval")
        expr = mod.body

    if expr.__class__ is ast.Name:
        name = expr.id
        if name in variables:
            if not getattr(variables[name], "constant", False):
                raise SyntaxError(f"Value {name} is not constant")
            if not getattr(variables[name], "scalar", False):
                raise SyntaxError(f"Value {name} is not scalar")
            return variables[name].get_value()
        elif name in ["True", "False"]:
            return 1.0 if name == "True" else 0.0
        else:
            raise ValueError(f"Unknown identifier {name}")
    elif expr.__class__ is ast.Constant:
        return expr.value
    elif expr.__class__ is ast.BoolOp:
        raise SyntaxError(
            "Cannot determine the numerical value for a boolean operation."
        )
    elif expr.__class__ is ast.Compare:
        raise SyntaxError(
            "Cannot determine the numerical value for a boolean operation."
        )
    elif expr.__class__ is ast.Call:
        raise SyntaxError("Cannot determine the numerical value for a function call.")
    elif expr.__class__ is ast.BinOp:
        op = expr.op.__class__.__name__
        left = _get_value_from_expression(expr.left, variables)
        right = _get_value_from_expression(expr.right, variables)
        if op == "Add" or op == "Sub":
            v = left + right
        elif op == "Mult":
            v = left * right
        elif op == "Div":
            v = float(left) / right
        elif op == "FloorDiv":
            v = left // right
        elif op == "Pow":
            v = left**right
        elif op == "Mod":
            v = left % right
        else:
            raise SyntaxError(f"Unsupported operation {op}")
        return v
    elif expr.__class__ is ast.UnaryOp:
        op = expr.op.__class__.__name__
        # check validity of operand and get its unit
        v = _get_value_from_expression(expr.operand, variables)
        if op == "Not":
            raise SyntaxError(
                "Cannot determine the numerical value for a boolean operation."
            )
        if op == "USub":
            return -v
        else:
            raise SyntaxError(f"Unknown unary operation {op}")
    else:
        raise SyntaxError(f"Unsupported operation {str(expr.__class__)}")


def parse_expression_dimensions(expr, variables, orig_expr=None):
    """
    Returns the unit value of an expression, and checks its validity

    Parameters
    ----------
    expr : str
        The expression to check.
    variables : dict
        Dictionary of all variables used in the `expr` (including `Constant`
        objects for external variables)

    Returns
    -------
    unit : Quantity
        The output unit of the expression

    Raises
    ------
    SyntaxError
        If the expression cannot be parsed, or if it uses ``a**b`` for ``b``
        anything other than a constant number.
    DimensionMismatchError
        If any part of the expression is dimensionally inconsistent.
    """

    # If we are working on a string, convert to the top level node
    if isinstance(expr, str):
        orig_expr = expr
        mod = ast.parse(expr, mode="eval")
        expr = mod.body
    if expr.__class__ is ast.Name:
        name = expr.id
        # Raise an error if a function is called as if it were a variable
        # (most of the time this happens for a TimedArray)
        if name in variables and isinstance(variables[name], Function):
            raise SyntaxError(
                f"{name} was used like a variable/constant, but it is a function.",
                ("<string>", expr.lineno, expr.col_offset + 1, orig_expr),
            )
        if name in variables:
            return get_dimensions(variables[name])
        elif name in ["True", "False"]:
            return DIMENSIONLESS
        else:
            raise KeyError(f"Unknown identifier {name}")
    elif expr.__class__ is ast.Constant:
        return DIMENSIONLESS
    elif expr.__class__ is ast.BoolOp:
        # check that the units are valid in each subexpression
        for node in expr.values:
            parse_expression_dimensions(node, variables, orig_expr=orig_expr)
        # but the result is a bool, so we just return 1 as the unit
        return DIMENSIONLESS
    elif expr.__class__ is ast.Compare:
        # check that the units are consistent in each subexpression
        subexprs = [expr.left] + expr.comparators
        subunits = []
        for node in subexprs:
            subunits.append(
                parse_expression_dimensions(node, variables, orig_expr=orig_expr)
            )
        for left_dim, right_dim in zip(subunits[:-1], subunits[1:]):
            if not have_same_dimensions(left_dim, right_dim):
                left_expr = NodeRenderer().render_node(expr.left)
                right_expr = NodeRenderer().render_node(expr.comparators[0])
                dim_left = get_dimensions(left_dim)
                dim_right = get_dimensions(right_dim)
                msg = (
                    "Comparison of expressions with different units. Expression "
                    f"'{left_expr}' has unit ({dim_left}), while expression "
                    f"'{right_expr}' has units ({dim_right})."
                )
                raise DimensionMismatchError(msg)
        # but the result is a bool, so we just return 1 as the unit
        return DIMENSIONLESS
    elif expr.__class__ is ast.Call:
        if len(expr.keywords):
            raise ValueError("Keyword arguments not supported.")
        elif getattr(expr, "starargs", None) is not None:
            raise ValueError("Variable number of arguments not supported")
        elif getattr(expr, "kwargs", None) is not None:
            raise ValueError("Keyword arguments not supported")

        func = variables.get(expr.func.id, None)
        if func is None:
            raise SyntaxError(
                f"Unknown function {expr.func.id}",
                ("<string>", expr.lineno, expr.col_offset + 1, orig_expr),
            )
        if not hasattr(func, "_arg_units") or not hasattr(func, "_return_unit"):
            raise ValueError(
                f"Function {expr.func.id} does not specify how it deals with units."
            )

        if len(func._arg_units) != len(expr.args):
            raise SyntaxError(
                f"Function '{expr.func.id}' was called with "
                f"{len(expr.args)} parameters, needs "
                f"{len(func._arg_units)}.",
                (
                    "<string>",
                    expr.lineno,
                    expr.col_offset + len(expr.func.id) + 1,
                    orig_expr,
                ),
            )

        for idx, (arg, expected_unit) in enumerate(zip(expr.args, func._arg_units)):
            arg_unit = parse_expression_dimensions(arg, variables, orig_expr=orig_expr)
            # A "None" in func._arg_units means: No matter what unit
            if expected_unit is None:
                continue
            # A string means: same unit as other argument
            elif isinstance(expected_unit, str):
                arg_idx = func._arg_names.index(expected_unit)
                expected_unit = parse_expression_dimensions(
                    expr.args[arg_idx], variables, orig_expr=orig_expr
                )
                if not have_same_dimensions(arg_unit, expected_unit):
                    msg = (
                        f"Argument number {idx + 1} for function "
                        f"{expr.func.id} was supposed to have the "
                        f"same units as argument number {arg_idx + 1}, but "
                        f"'{NodeRenderer().render_node(arg)}' has unit "
                        f"{get_unit_for_display(arg_unit)}, while "
                        f"'{NodeRenderer().render_node(expr.args[arg_idx])}' "
                        f"has unit {get_unit_for_display(expected_unit)}"
                    )
                    raise DimensionMismatchError(msg)
            elif expected_unit == bool:
                if not is_boolean_expression(arg, variables):
                    rendered_arg = NodeRenderer().render_node(arg)
                    raise TypeError(
                        f"Argument number {idx + 1} for function "
                        f"'{expr.func.id}' was expected to be a boolean "
                        f"value, but is '{rendered_arg}'."
                    )
            else:
                if not have_same_dimensions(arg_unit, expected_unit):
                    rendered_arg = NodeRenderer().render_node(arg)
                    arg_unit_dim = get_dimensions(arg_unit)
                    expected_unit_dim = get_dimensions(expected_unit)
                    msg = (
                        f"Argument number {idx+1} for function {expr.func.id} does "
                        f"not have the correct units. Expression '{rendered_arg}' "
                        f"has units ({arg_unit_dim}), but "
                        "should be "
                        f"({expected_unit_dim})."
                    )
                    raise DimensionMismatchError(msg)

        if func._return_unit == bool:
            return DIMENSIONLESS
        elif isinstance(func._return_unit, (Unit, int)):
            # Function always returns the same unit
            return getattr(func._return_unit, "dim", DIMENSIONLESS)
        else:
            # Function returns a unit that depends on the arguments
            arg_units = [
                parse_expression_dimensions(arg, variables, orig_expr=orig_expr)
                for arg in expr.args
            ]
            return func._return_unit(*arg_units).dim

    elif expr.__class__ is ast.BinOp:
        op = expr.op.__class__.__name__
        left_dim = parse_expression_dimensions(
            expr.left, variables, orig_expr=orig_expr
        )
        right_dim = parse_expression_dimensions(
            expr.right, variables, orig_expr=orig_expr
        )
        if op in ["Add", "Sub", "Mod"]:
            # dimensions should be the same
            if left_dim is not right_dim:
                op_symbol = {"Add": "+", "Sub": "-", "Mod": "%"}.get(op)
                left_str = NodeRenderer().render_node(expr.left)
                right_str = NodeRenderer().render_node(expr.right)
                left_unit = get_unit_for_display(left_dim)
                right_unit = get_unit_for_display(right_dim)
                error_msg = (
                    f"Expression '{left_str} {op_symbol} {right_str}' uses "
                    f"inconsistent units ('{left_str}' has unit "
                    f"{left_unit}; '{right_str}' "
                    f"has unit {right_unit})."
                )
                raise DimensionMismatchError(error_msg)
            u = left_dim
        elif op == "Mult":
            u = left_dim * right_dim
        elif op == "Div":
            u = left_dim / right_dim
        elif op == "FloorDiv":
            if not (left_dim is DIMENSIONLESS and right_dim is DIMENSIONLESS):
                if left_dim is DIMENSIONLESS:
                    col_offset = expr.right.col_offset + 1
                else:
                    col_offset = expr.left.col_offset + 1
                raise SyntaxError(
                    "Floor division can only be used on dimensionless values.",
                    ("<string>", expr.lineno, col_offset, orig_expr),
                )
            u = DIMENSIONLESS
        elif op == "Pow":
            if left_dim is DIMENSIONLESS and right_dim is DIMENSIONLESS:
                return DIMENSIONLESS
            n = _get_value_from_expression(expr.right, variables)
            u = left_dim**n
        else:
            raise SyntaxError(
                f"Unsupported operation {op}",
                (
                    "<string>",
                    expr.lineno,
                    getattr(
                        expr.left,
                        "end_col_offset",
                        len(NodeRenderer().render_node(expr.left)),
                    )
                    + 1,
                    orig_expr,
                ),
            )
        return u
    elif expr.__class__ is ast.UnaryOp:
        op = expr.op.__class__.__name__
        # check validity of operand and get its unit
        u = parse_expression_dimensions(expr.operand, variables, orig_expr=orig_expr)
        if op == "Not":
            return DIMENSIONLESS
        else:
            return u
    else:
        raise SyntaxError(
            f"Unsupported operation {str(expr.__class__.__name__)}",
            ("<string>", expr.lineno, expr.col_offset + 1, orig_expr),
        )
