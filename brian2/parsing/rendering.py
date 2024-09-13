import ast
import numbers

import numpy as np
import sympy

from brian2.core.functions import DEFAULT_CONSTANTS, DEFAULT_FUNCTIONS

__all__ = [
    "NodeRenderer",
    "NumpyNodeRenderer",
    "CPPNodeRenderer",
    "SympyNodeRenderer",
]


class NodeRenderer:
    expression_ops = {
        # BinOp
        "Add": "+",
        "Sub": "-",
        "Mult": "*",
        "Div": "/",
        "FloorDiv": "//",
        "Pow": "**",
        "Mod": "%",
        # Compare
        "Lt": "<",
        "LtE": "<=",
        "Gt": ">",
        "GtE": ">=",
        "Eq": "==",
        "NotEq": "!=",
        # Unary ops
        "Not": "not",
        "UAdd": "+",
        "USub": "-",
        # Bool ops
        "And": "and",
        "Or": "or",
        # Augmented assign
        "AugAdd": "+=",
        "AugSub": "-=",
        "AugMult": "*=",
        "AugDiv": "/=",
        "AugPow": "**=",
        "AugMod": "%=",
    }

    def __init__(self, auto_vectorise=None):
        if auto_vectorise is None:
            auto_vectorise = set()
        self.auto_vectorise = auto_vectorise

    def render_expr(self, expr, strip=True):
        if strip:
            expr = expr.strip()
        node = ast.parse(expr, mode="eval")
        return self.render_node(node.body)

    def render_node(self, node):
        nodename = node.__class__.__name__
        methname = f"render_{nodename}"
        try:
            return getattr(self, methname)(node)
        except AttributeError:
            if nodename == "Subscript":
                raise SyntaxError(
                    "Brian equations/expressions do not support indexing with '[...]'."
                )
            elif nodename == "Attribute":
                raise SyntaxError(
                    "Brian equations/expressions do not support accessing attributes"
                    " with the '.' syntax."
                )
            elif nodename == "Tuple":
                raise SyntaxError("Brian equations/expressions do not support tuples.")
            else:
                raise SyntaxError(
                    f"Brian equations/expressions do not support the '{nodename}'"
                    " syntax."
                )

    def render_func(self, node):
        return self.render_Name(node)

    def render_Name(self, node):
        return node.id

    def render_Constant(self, node):
        if isinstance(node.value, np.number):
            # repr prints the dtype in numpy 2.0
            return repr(node.value.item())
        return repr(node.value)

    def render_Call(self, node):
        if len(node.keywords):
            raise ValueError("Keyword arguments not supported.")
        elif getattr(node, "starargs", None) is not None:
            raise ValueError("Variable number of arguments not supported")
        elif getattr(node, "kwargs", None) is not None:
            raise ValueError("Keyword arguments not supported")
        else:
            if node.func.id in self.auto_vectorise:
                vectorisation_idx = ast.Name("_vectorisation_idx")
                args = node.args + [vectorisation_idx]
            else:
                args = node.args
            return f"{self.render_func(node.func)}({', '.join(self.render_node(arg) for arg in args)})"

    def render_element_parentheses(self, node):
        """
        Render an element with parentheses around it or leave them away for
        numbers, names and function calls.
        """
        if node.__class__.__name__ == "Name":
            return self.render_node(node)
        elif node.__class__.__name__ in ["Num", "Constant"] and node.value >= 0:
            return self.render_node(node)
        elif node.__class__.__name__ == "Call":
            return self.render_node(node)
        else:
            return f"({self.render_node(node)})"

    def render_BinOp_parentheses(self, left, right, op):
        # Use a simplified checking whether it is possible to omit parentheses:
        # only omit parentheses for numbers, variable names or function calls.
        # This means we still put needless parentheses because we ignore
        # precedence rules, e.g. we write "3 + (4 * 5)" but at least we do
        # not do "(3) + ((4) + (5))"
        op_class = op.__class__.__name__
        # Give a more useful error message when using bit-wise operators
        if op_class in ["BitXor", "BitAnd", "BitOr"]:
            correction = {
                "BitXor": ("^", "**"),
                "BitAnd": ("&", "and"),
                "BitOr": ("|", "or"),
            }.get(op_class)
            raise SyntaxError(
                f'The operator "{correction[0]}" is not supported, use'
                f' "{correction[1]}" instead.'
            )
        return (
            f"{self.render_element_parentheses(left)} "
            f"{self.expression_ops[op_class]} "
            f"{self.render_element_parentheses(right)}"
        )

    def render_BinOp(self, node):
        return self.render_BinOp_parentheses(node.left, node.right, node.op)

    def render_BoolOp(self, node):
        op = self.expression_ops[node.op.__class__.__name__]
        return (f" {op} ").join(
            f"{self.render_element_parentheses(v)}" for v in node.values
        )

    def render_Compare(self, node):
        if len(node.comparators) > 1:
            raise SyntaxError("Can only handle single comparisons like a<b not a<b<c")
        return self.render_BinOp_parentheses(
            node.left, node.comparators[0], node.ops[0]
        )

    def render_UnaryOp(self, node):
        return f"{self.expression_ops[node.op.__class__.__name__]} {self.render_element_parentheses(node.operand)}"

    def render_Assign(self, node):
        if len(node.targets) > 1:
            raise SyntaxError("Only support syntax like a=b not a=b=c")
        return f"{self.render_node(node.targets[0])} = {self.render_node(node.value)}"

    def render_AugAssign(self, node):
        target = node.target.id
        rhs = self.render_node(node.value)
        op = self.expression_ops[f"Aug{node.op.__class__.__name__}"]
        return f"{target} {op} {rhs}"


class NumpyNodeRenderer(NodeRenderer):
    expression_ops = NodeRenderer.expression_ops.copy()
    expression_ops.update(
        {
            # Unary ops
            # We'll handle "not" explicitly below
            # Bool ops
            "And": "&",
            "Or": "|",
        }
    )

    def render_UnaryOp(self, node):
        if node.op.__class__.__name__ == "Not":
            return f"logical_not({self.render_node(node.operand)})"
        else:
            return NodeRenderer.render_UnaryOp(self, node)


class SympyNodeRenderer(NodeRenderer):
    expression_ops = {
        "Add": sympy.Add,
        "Mult": sympy.Mul,
        "Pow": sympy.Pow,
        "Mod": sympy.Mod,
        # Compare
        "Lt": sympy.StrictLessThan,
        "LtE": sympy.LessThan,
        "Gt": sympy.StrictGreaterThan,
        "GtE": sympy.GreaterThan,
        "Eq": sympy.Eq,
        "NotEq": sympy.Ne,
        # Unary ops are handled manually
        # Bool ops
        "And": sympy.And,
        "Or": sympy.Or,
    }

    def render_func(self, node):
        if node.id in DEFAULT_FUNCTIONS:
            f = DEFAULT_FUNCTIONS[node.id]
            if f.sympy_func is not None and isinstance(
                f.sympy_func, sympy.FunctionClass
            ):
                return f.sympy_func
        # special workaround for the "int" function
        if node.id == "int":
            return sympy.Function("int_")
        else:
            return sympy.Function(node.id)

    def render_Call(self, node):
        if len(node.keywords):
            raise ValueError("Keyword arguments not supported.")
        elif getattr(node, "starargs", None) is not None:
            raise ValueError("Variable number of arguments not supported")
        elif getattr(node, "kwargs", None) is not None:
            raise ValueError("Keyword arguments not supported")
        elif len(node.args) == 0:
            return self.render_func(node.func)(sympy.Symbol("_placeholder_arg"))
        else:
            return self.render_func(node.func)(
                *(self.render_node(arg) for arg in node.args)
            )

    def render_Compare(self, node):
        if len(node.comparators) > 1:
            raise SyntaxError("Can only handle single comparisons like a<b not a<b<c")
        op = node.ops[0]
        return self.expression_ops[op.__class__.__name__](
            self.render_node(node.left), self.render_node(node.comparators[0])
        )

    def render_Name(self, node):
        if node.id in DEFAULT_CONSTANTS:
            c = DEFAULT_CONSTANTS[node.id]
            return c.sympy_obj
        elif node.id in ["t", "dt"]:
            return sympy.Symbol(node.id, real=True, positive=True)
        else:
            return sympy.Symbol(node.id, real=True)

    def render_Constant(self, node):
        if node.value is True or node.value is False:
            return node.value
        elif isinstance(node.value, numbers.Integral):
            return sympy.Integer(node.value)
        elif isinstance(node.value, numbers.Number):
            return sympy.Float(node.value)
        else:
            return str(node.value)

    def render_BinOp(self, node):
        op_name = node.op.__class__.__name__
        # Sympy implements division and subtraction as multiplication/addition
        if op_name == "Div":
            op = self.expression_ops["Mult"]
            return op(self.render_node(node.left), 1 / self.render_node(node.right))
        elif op_name == "FloorDiv":
            op = self.expression_ops["Mult"]
            left = self.render_node(node.left)
            right = self.render_node(node.right)
            return sympy.floor(op(left, 1 / right))
        elif op_name == "Sub":
            op = self.expression_ops["Add"]
            return op(self.render_node(node.left), -self.render_node(node.right))
        else:
            op = self.expression_ops[op_name]
            return op(self.render_node(node.left), self.render_node(node.right))

    def render_BoolOp(self, node):
        op = self.expression_ops[node.op.__class__.__name__]
        return op(*(self.render_node(value) for value in node.values))

    def render_UnaryOp(self, node):
        op_name = node.op.__class__.__name__
        if op_name == "UAdd":
            # Nothing to do
            return self.render_node(node.operand)
        elif op_name == "USub":
            return -self.render_node(node.operand)
        elif op_name == "Not":
            return sympy.Not(self.render_node(node.operand))
        else:
            raise ValueError(f"Unknown unary operator: {op_name}")


class CPPNodeRenderer(NodeRenderer):
    expression_ops = NodeRenderer.expression_ops.copy()
    expression_ops.update(
        {
            # Unary ops
            "Not": "!",
            # Bool ops
            "And": "&&",
            "Or": "||",
            # C does not have a floor division operator (but see render_BinOp)
            "FloorDiv": "/",
        }
    )

    def render_BinOp(self, node):
        if node.op.__class__.__name__ == "Pow":
            return (
                f"_brian_pow({self.render_node(node.left)},"
                f" {self.render_node(node.right)})"
            )
        elif node.op.__class__.__name__ == "Mod":
            return (
                f"_brian_mod({self.render_node(node.left)},"
                f" {self.render_node(node.right)})"
            )
        elif node.op.__class__.__name__ == "Div":
            # C uses integer division, this is a quick and dirty way to assure
            # it uses floating point division for integers
            return f"1.0f*{self.render_element_parentheses(node.left)}/{self.render_element_parentheses(node.right)}"
        elif node.op.__class__.__name__ == "FloorDiv":
            return (
                f"_brian_floordiv({self.render_node(node.left)},"
                f" {self.render_node(node.right)})"
            )
        else:
            return NodeRenderer.render_BinOp(self, node)

    def render_Constant(self, node):
        if node.value is True:
            return "true"
        elif node.value is False:
            return "false"
        else:
            return super().render_Constant(node)

    def render_Name(self, node):
        if node.id == "inf":
            return "INFINITY"
        else:
            return node.id

    def render_Assign(self, node):
        return f"{NodeRenderer.render_Assign(self, node)};"
