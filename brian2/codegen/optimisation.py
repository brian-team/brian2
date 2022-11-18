"""
Simplify and optimise sequences of statements by rewriting and pulling out loop invariants.
"""

import ast
from collections import OrderedDict
import copy
import itertools
from functools import reduce

from brian2.core.functions import DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS
from brian2.core.variables import AuxiliaryVariable
from brian2.parsing.bast import (
    brian_ast,
    BrianASTRenderer,
    dtype_hierarchy,
    brian_dtype_from_dtype,
)
from brian2.parsing.rendering import NodeRenderer, get_node_value
from brian2.utils.stringtools import get_identifiers, word_substitute
from brian2.core.preferences import prefs

from .statements import Statement

# Default namespace has all the standard functions and constants in it
defaults_ns = dict((k, v.pyfunc) for k, v in DEFAULT_FUNCTIONS.items())
defaults_ns.update(dict((k, v.value) for k, v in DEFAULT_CONSTANTS.items()))


__all__ = ["optimise_statements", "ArithmeticSimplifier", "Simplifier"]


def evaluate_expr(expr, ns):
    """
    Try to evaluate the expression in the given namespace

    Returns either (value, True) if successful, or (expr, False) otherwise.

    Example
    -------
    >>> assumptions = {'sin': DEFAULT_FUNCTIONS['sin'].pyfunc,
    ...                'pi': DEFAULT_CONSTANTS['pi'].value}
    >>> evaluate_expr('1/2', assumptions)
    (0.5, True)
    >>> evaluate_expr('sin(pi/2)', assumptions)
    (1.0, True)
    >>> evaluate_expr('sin(2*pi*freq*t)', assumptions)
    ('sin(2*pi*freq*t)', False)
    >>> evaluate_expr('1/0', assumptions)
    ('1/0', False)
    """
    try:
        val = eval(expr, ns)
        return val, True
    except (NameError, ArithmeticError):
        return expr, False


def expression_complexity(expr, variables):
    return brian_ast(expr, variables).complexity


def optimise_statements(scalar_statements, vector_statements, variables, blockname=""):
    """
    Optimise a sequence of scalar and vector statements

    Performs the following optimisations:

    1. Constant evaluations (e.g. exp(0) to 1). See `evaluate_expr`.
    2. Arithmetic simplifications (e.g. 0*x to 0). See `ArithmeticSimplifier`, `collect`.
    3. Pulling out loop invariants (e.g. v*exp(-dt/tau) to a=exp(-dt/tau) outside the loop and v*a inside).
       See `Simplifier`.
    4. Boolean simplifications (allowing the replacement of expressions with booleans with a sequence of if/thens).
       See `Simplifier`.

    Parameters
    ----------
    scalar_statements : sequence of Statement
        Statements that only involve scalar values and should be evaluated in the scalar block.
    vector_statements : sequence of Statement
        Statements that involve vector values and should be evaluated in the vector block.
    variables : dict of (str, Variable)
        Definition of the types of the variables.
    blockname : str, optional
        Name of the block (used for LIO constant prefixes to avoid name clashes)

    Returns
    -------
    new_scalar_statements : sequence of Statement
        As above but with loop invariants pulled out from vector statements
    new_vector_statements : sequence of Statement
        Simplified/optimised versions of statements
    """
    boolvars = dict(
        (k, v)
        for k, v in variables.items()
        if hasattr(v, "dtype") and brian_dtype_from_dtype(v.dtype) == "boolean"
    )
    # We use the Simplifier class by rendering each expression, which generates new scalar statements
    # stored in the Simplifier object, and these are then added to the scalar statements.
    simplifier = Simplifier(variables, scalar_statements, extra_lio_prefix=blockname)
    new_vector_statements = []
    for stmt in vector_statements:
        # Carry out constant evaluation, arithmetic simplification and loop invariants
        new_expr = simplifier.render_expr(stmt.expr)
        new_stmt = Statement(
            stmt.var,
            stmt.op,
            new_expr,
            stmt.comment,
            dtype=stmt.dtype,
            constant=stmt.constant,
            subexpression=stmt.subexpression,
            scalar=stmt.scalar,
        )
        # Now check if boolean simplification can be carried out
        complexity_std = expression_complexity(new_expr, simplifier.variables)
        idents = get_identifiers(new_expr)
        used_boolvars = [var for var in boolvars if var in idents]
        if len(used_boolvars):
            # We want to iterate over all the possible assignments of boolean variables to values in (True, False)
            bool_space = [[False, True] for _ in used_boolvars]
            expanded_expressions = {}
            complexities = {}
            for bool_vals in itertools.product(*bool_space):
                # substitute those values into the expr and simplify (including potentially pulling out new
                # loop invariants)
                subs = dict(
                    (var, str(val)) for var, val in zip(used_boolvars, bool_vals)
                )
                curexpr = word_substitute(new_expr, subs)
                curexpr = simplifier.render_expr(curexpr)
                key = tuple((var, val) for var, val in zip(used_boolvars, bool_vals))
                expanded_expressions[key] = curexpr
                complexities[key] = expression_complexity(curexpr, simplifier.variables)
            # See Statement for details on these
            new_stmt.used_boolean_variables = used_boolvars
            new_stmt.boolean_simplified_expressions = expanded_expressions
            new_stmt.complexity_std = complexity_std
            new_stmt.complexities = complexities
        new_vector_statements.append(new_stmt)
    # Generate additional scalar statements for the loop invariants
    new_scalar_statements = copy.copy(scalar_statements)
    for expr, name in simplifier.loop_invariants.items():
        dtype_name = simplifier.loop_invariant_dtypes[name]
        if dtype_name == "boolean":
            dtype = bool
        elif dtype_name == "integer":
            dtype = int
        else:
            dtype = prefs.core.default_float_dtype
        new_stmt = Statement(
            name,
            ":=",
            expr,
            "",
            dtype=dtype,
            constant=True,
            subexpression=False,
            scalar=True,
        )
        new_scalar_statements.append(new_stmt)
    return new_scalar_statements, new_vector_statements


def _replace_with_zero(zero_node, node):
    """
    Helper function to return a "zero node" of the correct type.

    Parameters
    ----------
    zero_node : `ast.Num`
        The node to replace
    node : `ast.Node`
        The node that determines the type

    Returns
    -------
    zero_node : `ast.Num`
        The original ``zero_node`` with its value replaced by 0 or 0.0.
    """
    # must not change the dtype of the output,
    # e.g. handle 0/float->0.0 and 0.0/int->0.0
    zero_node.dtype = node.dtype
    if node.dtype == "integer":
        zero_node.value = 0
    else:
        zero_node.value = prefs.core.default_float_dtype(0.0)
    return zero_node


class ArithmeticSimplifier(BrianASTRenderer):
    """
    Carries out the following arithmetic simplifications:

    1. Constant evaluation (e.g. exp(0)=1) by attempting to evaluate the expression in an "assumptions namespace"
    2. Binary operators, e.g. 0*x=0, 1*x=x, etc. You have to take care that the dtypes match here, e.g.
       if x is an integer, then 1.0*x shouldn't be replaced with x but left as 1.0*x.

    Parameters
    ----------
    variables : dict of (str, Variable)
        Usual definition of variables.
    assumptions : sequence of str
        Additional assumptions that can be used in simplification, each assumption is a string statement.
        These might be the scalar statements for example.
    """

    def __init__(self, variables):
        BrianASTRenderer.__init__(self, variables, copy_variables=False)
        self.assumptions = []
        self.assumptions_ns = dict(defaults_ns)
        self.bast_renderer = BrianASTRenderer(variables, copy_variables=False)

    def render_node(self, node):
        """
        Assumes that the node has already been fully processed by BrianASTRenderer
        """
        if not hasattr(node, "simplified"):
            node = super(ArithmeticSimplifier, self).render_node(node)
            node.simplified = True
        # can't evaluate vector expressions, so abandon in this case
        if not node.scalar:
            return node
        # No evaluation necessary for simple names or numbers
        if node.__class__.__name__ in ["Name", "NameConstant", "Num", "Constant"]:
            return node
        # Don't evaluate stateful nodes (e.g. those containing a rand() call)
        if not node.stateless:
            return node
        # try fully evaluating using assumptions
        expr = NodeRenderer().render_node(node)
        val, evaluated = evaluate_expr(expr, self.assumptions_ns)
        if evaluated:
            if node.dtype == "boolean":
                val = bool(val)
                if hasattr(ast, "Constant"):
                    newnode = ast.Constant(val)
                elif hasattr(ast, "NameConstant"):
                    newnode = ast.NameConstant(val)
                else:
                    # None is the expression context, we don't use it so we just set to None
                    newnode = ast.Name(repr(val), None)
            elif node.dtype == "integer":
                val = int(val)
            else:
                val = prefs.core.default_float_dtype(val)
            if node.dtype != "boolean":
                if hasattr(ast, "Constant"):
                    newnode = ast.Constant(val)
                else:
                    newnode = ast.Num(val)
            newnode.dtype = node.dtype
            newnode.scalar = True
            newnode.stateless = node.stateless
            newnode.complexity = 0
            return newnode
        return node

    def render_BinOp(self, node):
        if node.dtype == "float":  # only try to collect float type nodes
            if node.op.__class__.__name__ in [
                "Mult",
                "Div",
                "Add",
                "Sub",
            ] and not hasattr(node, "collected"):
                newnode = self.bast_renderer.render_node(collect(node))
                newnode.collected = True
                return self.render_node(newnode)
        left = node.left = self.render_node(node.left)
        right = node.right = self.render_node(node.right)
        node = super(ArithmeticSimplifier, self).render_BinOp(node)
        op = node.op
        # Handle multiplication by 0 or 1
        if op.__class__.__name__ == "Mult":
            for operand, other in [(left, right), (right, left)]:
                if operand.__class__.__name__ in ["Num", "Constant"]:
                    op_value = get_node_value(operand)
                    if op_value == 0:
                        # Do not remove stateful functions
                        if node.stateless:
                            return _replace_with_zero(operand, node)
                    if op_value == 1:
                        # only simplify this if the type wouldn't be cast by the operation
                        if (
                            dtype_hierarchy[operand.dtype]
                            <= dtype_hierarchy[other.dtype]
                        ):
                            return other
        # Handle division by 1, or 0/x
        elif op.__class__.__name__ == "Div":
            if (
                left.__class__.__name__ in ["Num", "Constant"]
                and get_node_value(left) == 0
            ):  # 0/x
                if node.stateless:
                    # Do not remove stateful functions
                    return _replace_with_zero(left, node)
            if (
                right.__class__.__name__ in ["Num", "Constant"]
                and get_node_value(right) == 1
            ):  # x/1
                # only simplify this if the type wouldn't be cast by the operation
                if dtype_hierarchy[right.dtype] <= dtype_hierarchy[left.dtype]:
                    return left
        elif op.__class__.__name__ == "FloorDiv":
            if (
                left.__class__.__name__ in ["Num", "Constant"]
                and get_node_value(left) == 0
            ):  # 0//x
                if node.stateless:
                    # Do not remove stateful functions
                    return _replace_with_zero(left, node)
            # Only optimise floor division by 1 if both numbers are integers,
            # for floating point values, floor division by 1 changes the value,
            # and division by 1.0 can change the type for an integer value
            if (
                left.dtype == right.dtype == "integer"
                and right.__class__.__name__ in ["Num", "Constant"]
                and get_node_value(right) == 1
            ):  # x//1
                return left
        # Handle addition of 0
        elif op.__class__.__name__ == "Add":
            for operand, other in [(left, right), (right, left)]:
                if (
                    operand.__class__.__name__ in ["Num", "Constant"]
                    and get_node_value(operand) == 0
                ):
                    # only simplify this if the type wouldn't be cast by the operation
                    if dtype_hierarchy[operand.dtype] <= dtype_hierarchy[other.dtype]:
                        return other
        # Handle subtraction of 0
        elif op.__class__.__name__ == "Sub":
            if (
                right.__class__.__name__ in ["Num", "Constant"]
                and get_node_value(right) == 0
            ):
                # only simplify this if the type wouldn't be cast by the operation
                if dtype_hierarchy[right.dtype] <= dtype_hierarchy[left.dtype]:
                    return left

        # simplify e.g. 2*float to 2.0*float to make things more explicit: not strictly necessary
        # but might be useful for some codegen targets
        if node.dtype == "float" and op.__class__.__name__ in [
            "Mult",
            "Add",
            "Sub",
            "Div",
        ]:
            for subnode in [node.left, node.right]:
                if subnode.__class__.__name__ in ["Num", "Constant"] and not (
                    get_node_value(subnode) is True or get_node_value(subnode) is False
                ):
                    subnode.dtype = "float"
                    subnode.value = prefs.core.default_float_dtype(
                        get_node_value(subnode)
                    )
        return node


class Simplifier(BrianASTRenderer):
    """
    Carry out arithmetic simplifications (see `ArithmeticSimplifier`) and loop invariants

    Parameters
    ----------
    variables : dict of (str, Variable)
        Usual definition of variables.
    scalar_statements : sequence of Statement
        Predefined scalar statements that can be used as part of simplification

    Notes
    -----

    After calling `render_expr` on a sequence of expressions (coming from vector statements typically),
    this object will have some new attributes:

    ``loop_invariants`` : OrderedDict of (expression, varname)
        varname will be of the form ``_lio_N`` where ``N`` is some integer, and the expressions will be
        strings that correspond to scalar-only expressions that can be evaluated outside of the vector
        block.
    ``loop_invariant_dtypes`` : dict of (varname, dtypename)
        dtypename will be one of ``'boolean'``, ``'integer'``, ``'float'``.
    """

    def __init__(self, variables, scalar_statements, extra_lio_prefix=""):
        BrianASTRenderer.__init__(self, variables, copy_variables=False)
        self.loop_invariants = OrderedDict()
        self.loop_invariant_dtypes = {}
        self.value = 0
        self.node_renderer = NodeRenderer()
        self.arithmetic_simplifier = ArithmeticSimplifier(variables)
        self.scalar_statements = scalar_statements
        if extra_lio_prefix is None:
            extra_lio_prefix = ""
        if len(extra_lio_prefix):
            extra_lio_prefix = f"{extra_lio_prefix}_"
        self.extra_lio_prefix = extra_lio_prefix

    def render_expr(self, expr):
        node = brian_ast(expr, self.variables)
        node = self.arithmetic_simplifier.render_node(node)
        node = self.render_node(node)
        return self.node_renderer.render_node(node)

    def render_node(self, node):
        """
        Assumes that the node has already been fully processed by BrianASTRenderer
        """
        # can we pull this out?
        if node.scalar and node.complexity > 0:
            expr = self.node_renderer.render_node(
                self.arithmetic_simplifier.render_node(node)
            )
            if expr in self.loop_invariants:
                name = self.loop_invariants[expr]
            else:
                self.value += 1
                name = f"_lio_{self.extra_lio_prefix}{str(self.value)}"
                self.loop_invariants[expr] = name
                self.loop_invariant_dtypes[name] = node.dtype
                numpy_dtype = {
                    "boolean": bool,
                    "integer": int,
                    "float": prefs.core.default_float_dtype,
                }[node.dtype]
                self.variables[name] = AuxiliaryVariable(
                    name, dtype=numpy_dtype, scalar=True
                )
            # None is the expression context, we don't use it so we just set to None
            newnode = ast.Name(name, None)
            newnode.scalar = True
            newnode.dtype = node.dtype
            newnode.complexity = 0
            newnode.stateless = node.stateless
            return newnode
        # otherwise, render node as usual
        return super(Simplifier, self).render_node(node)


def reduced_node(terms, op):
    """
    Reduce a sequence of terms with the given operator

    For examples, if terms were [a, b, c] and op was multiplication then the reduction would be (a*b)*c.

    Parameters
    ----------
    terms : list
        AST nodes.
    op : AST node
        Could be `ast.Mult` or `ast.Add`.

    Examples
    --------
    >>> import ast
    >>> nodes = [ast.Name(id='x'), ast.Name(id='y'), ast.Name(id='z')]
    >>> ast.dump(reduced_node(nodes, ast.Mult), annotate_fields=False)
    "BinOp(BinOp(Name('x'), Mult(), Name('y')), Mult(), Name('z'))"
    >>> nodes = [ast.Name(id='x')]
    >>> ast.dump(reduced_node(nodes, ast.Add), annotate_fields=False)
    "Name('x')"
    """
    # Remove None terms
    terms = [term for term in terms if term is not None]
    if not len(terms):
        return None
    return reduce(lambda left, right: ast.BinOp(left, op(), right), terms)


def cancel_identical_terms(primary, inverted):
    """
    Cancel terms in a collection, e.g. a+b-a should be cancelled to b

    Simply renders the nodes into expressions and removes whenever there is a common expression
    in primary and inverted.

    Parameters
    ----------
    primary : list of AST nodes
        These are the nodes that are positive with respect to the operator, e.g.
        in x*y/z it would be [x, y].
    inverted : list of AST nodes
        These are the nodes that are inverted with respect to the operator, e.g.
        in x*y/z it would be [z].

    Returns
    -------
    primary : list of AST nodes
        Primary nodes after cancellation
    inverted : list of AST nodes
        Inverted nodes after cancellation
    """
    nr = NodeRenderer()
    expressions = dict((node, nr.render_node(node)) for node in primary)
    expressions.update(dict((node, nr.render_node(node)) for node in inverted))
    new_primary = []
    inverted_expressions = [expressions[term] for term in inverted]
    for term in primary:
        expr = expressions[term]
        if expr in inverted_expressions and term.stateless:
            new_inverted = []
            for iterm in inverted:
                if expressions[iterm] == expr:
                    expr = ""  # handled
                else:
                    new_inverted.append(iterm)
            inverted = new_inverted
            inverted_expressions = [expressions[term] for term in inverted]
        else:
            new_primary.append(term)
    return new_primary, inverted


def collect(node):
    """
    Attempts to collect commutative operations into one and simplifies them.

    For example, if x and y are scalars, and z is a vector, then (x*z)*y should
    be rewritten as (x*y)*z to minimise the number of vector operations. Similarly,
    ((x*2)*3)*4 should be rewritten as x*24.

    Works for either multiplication/division or addition/subtraction nodes.

    The final output is a subexpression of the following maximal form:

        (((numerical_value*(product of scalars))/(product of scalars))*(product of vectors))/(product of vectors)

    Any possible cancellations will have been done.

    Parameters
    ----------
    node : Brian AST node
        The node to be collected/simplified.

    Returns
    -------
    node : Brian AST node
        Simplified node.
    """
    node.collected = True
    orignode_dtype = node.dtype
    # we only work on */ or +- ops, which are both BinOp
    if node.__class__.__name__ != "BinOp":
        return node
    # primary would be the * or + nodes, and inverted would be the / or - nodes
    terms_primary = []
    terms_inverted = []
    # we handle both multiplicative and additive nodes in the same way by using these variables
    if node.op.__class__.__name__ in ["Mult", "Div"]:
        op_primary = ast.Mult
        op_inverted = ast.Div
        op_null = prefs.core.default_float_dtype(1.0)  # the identity for the operator
        op_py_primary = lambda x, y: x * y
        op_py_inverted = lambda x, y: x / y
    elif node.op.__class__.__name__ in ["Add", "Sub"]:
        op_primary = ast.Add
        op_inverted = ast.Sub
        op_null = prefs.core.default_float_dtype(0.0)
        op_py_primary = lambda x, y: x + y
        op_py_inverted = lambda x, y: x - y
    else:
        return node
    if node.dtype == "integer":
        op_null_with_dtype = int(op_null)
    else:
        op_null_with_dtype = op_null
    # recursively collect terms into the terms_primary and terms_inverted lists
    collect_commutative(node, op_primary, op_inverted, terms_primary, terms_inverted)
    x = op_null
    # extract the numerical nodes and fully evaluate
    remaining_terms_primary = []
    remaining_terms_inverted = []
    for term in terms_primary:
        if term.__class__.__name__ == "Num":
            x = op_py_primary(x, term.n)
        elif term.__class__.__name__ == "Constant":
            x = op_py_primary(x, term.value)
        else:
            remaining_terms_primary.append(term)
    for term in terms_inverted:
        if term.__class__.__name__ == "Num":
            x = op_py_inverted(x, term.n)
        elif term.__class__.__name__ == "Constant":
            x = op_py_inverted(x, term.value)
        else:
            remaining_terms_inverted.append(term)
    # if the fully evaluated node is just the identity/null element then we
    # don't have to make it into an explicit term
    if x != op_null:
        if hasattr(ast, "Constant"):
            num_node = ast.Constant(x)
        else:
            num_node = ast.Num(x)
    else:
        num_node = None
    terms_primary = remaining_terms_primary
    terms_inverted = remaining_terms_inverted
    node = num_node
    for scalar in (True, False):
        primary_terms = [term for term in terms_primary if term.scalar == scalar]
        inverted_terms = [term for term in terms_inverted if term.scalar == scalar]
        primary_terms, inverted_terms = cancel_identical_terms(
            primary_terms, inverted_terms
        )

        # produce nodes that are the reduction of the operator on these subsets
        prod_primary = reduced_node(primary_terms, op_primary)
        prod_inverted = reduced_node(inverted_terms, op_primary)

        # construct the simplest version of the fully simplified node (only doing operations where necessary)
        node = reduced_node([node, prod_primary], op_primary)
        if prod_inverted is not None:
            if node is None:
                if hasattr(ast, "Constant"):
                    node = ast.Constant(op_null_with_dtype)
                else:
                    node = ast.Num(op_null_with_dtype)
            node = ast.BinOp(node, op_inverted(), prod_inverted)

    if node is None:  # everything cancelled
        if hasattr(ast, "Constant"):
            node = ast.Constant(op_null_with_dtype)
        else:
            node = ast.Num(op_null_with_dtype)
    if (
        hasattr(node, "dtype")
        and dtype_hierarchy[node.dtype] < dtype_hierarchy[orignode_dtype]
    ):
        node = ast.BinOp(ast.Num(op_null_with_dtype), op_primary(), node)
    node.collected = True
    return node


def collect_commutative(
    node, primary, inverted, terms_primary, terms_inverted, add_to_inverted=False
):
    # This function is called recursively, so we use add_to_inverted to keep track of whether or not
    # we're working in the numerator/denominator (for multiplicative nodes, equivalent for additive).
    op_primary = node.op.__class__ is primary
    # this should only be called with node a BinOp of type primary or inverted
    # left_exact is the condition that we can collect terms (we can do it with floats or add/sub,
    # but not integer mult/div - the reason being that for C-style division e.g. 3/(4/3)!=(3*3)/4
    left_exact = node.left.dtype == "float" or (
        hasattr(node.left, "op") and node.left.op.__class__.__name__ in ["Add", "Sub"]
    )
    if (
        node.left.__class__.__name__ == "BinOp"
        and node.left.op.__class__ in [primary, inverted]
        and left_exact
    ):
        collect_commutative(
            node.left,
            primary,
            inverted,
            terms_primary,
            terms_inverted,
            add_to_inverted=add_to_inverted,
        )
    else:
        if add_to_inverted:
            terms_inverted.append(node.left)
        else:
            terms_primary.append(node.left)
    right_exact = node.right.dtype == "float" or (
        hasattr(node.right, "op") and node.right.op.__class__.__name__ in ["Add", "Sub"]
    )
    if (
        node.right.__class__.__name__ == "BinOp"
        and node.right.op.__class__ in [primary, inverted]
        and right_exact
    ):
        if node.op.__class__ is primary:
            collect_commutative(
                node.right,
                primary,
                inverted,
                terms_primary,
                terms_inverted,
                add_to_inverted=add_to_inverted,
            )
        else:
            collect_commutative(
                node.right,
                primary,
                inverted,
                terms_primary,
                terms_inverted,
                add_to_inverted=not add_to_inverted,
            )
    else:
        if (not add_to_inverted and op_primary) or (add_to_inverted and not op_primary):
            terms_primary.append(node.right)
        else:
            terms_inverted.append(node.right)
