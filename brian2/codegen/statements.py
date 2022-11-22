"""
Module providing the `Statement` class.
"""


class Statement(object):
    """
    A single line mathematical statement.

    The structure is ``var op expr``.

    Parameters
    ----------
    var : str
        The left hand side of the statement, the value being written to.
    op : str
        The operation, can be any of the standard Python operators (including
        ``+=`` etc.) or a special operator ``:=`` which means you are defining
        a new symbol (whereas ``=`` means you are setting the value of an
        existing symbol).
    expr : str, `Expression`
        The right hand side of the statement.
    dtype : `dtype`
        The numpy dtype of the value or array `var`.
    constant : bool, optional
        Set this flag to ``True`` if the value will not change (only applies for
        ``op==':='``.
    subexpression : bool, optional
        Set this flag to ``True`` if the variable is a subexpression. In some
        languages (e.g. Python) you can use this to save a memory copy, because
        you don't need to do ``lhs[:] = rhs`` but a redefinition ``lhs = rhs``.
    scalar : bool, optional
        Set this flag to ``True`` if `var` and `expr` are scalar.

    Notes
    -----
    Will compute the following attribute:

    ``inplace``
        True or False depending if the operation is in-place or not.

    Boolean simplification notes:

    Will initially set the attribute ``used_boolean_variables`` to ``None``.
    This is set by `~brian2.codegen.optimisation.optimise_statements` when it
    is called on a sequence of statements to the list of boolean variables
    that are used in this expression. In addition, the attribute
    ``boolean_simplified_expressions`` is set to a dictionary with keys
    consisting of a tuple of pairs ``(var, value)`` where ``var`` is the
    name of the boolean variable (will be in ``used_boolean_variables``)
    and ``var`` is ``True`` or ``False``. The values of the dictionary are
    strings representing the simplified version of the expression if each
    ``var=value`` substitution is made for that key. The keys will range
    over all possible values of the set of boolean variables. The complexity
    of the original statement is set as the attribute ``complexity_std``,
    and the complexity of the simplified versions are in the dictionary
    ``complexities`` (with the same keys).

    This information can be used to generate code that replaces a complex
    expression that varies depending on the value of one or more boolean
    variables with an ``if/then`` sequence where each subexpression is
    simplified. It is optional to use this (e.g. the numpy codegen does
    not, but the cython one does).
    """

    def __init__(
        self,
        var,
        op,
        expr,
        comment,
        dtype,
        constant=False,
        subexpression=False,
        scalar=False,
    ):
        self.var = var.strip()
        self.op = op.strip()
        self.expr = expr
        self.comment = comment
        self.dtype = dtype
        self.constant = constant
        self.subexpression = subexpression
        self.scalar = scalar
        if constant and self.op != ":=":
            raise ValueError(f"Should not set constant flag for operation {self.op}")
        if op.endswith("=") and op != "=" and op != ":=":
            self.inplace = True
        else:
            self.inplace = False
        self.used_boolean_variables = None
        self.boolean_simplified_expressions = None

    def __str__(self):
        s = f"{self.var} {self.op} {str(self.expr)}"
        if self.constant:
            s += " (constant)"
        if self.subexpression:
            s += " (subexpression)"
        if self.inplace:
            s += " (in-place)"
        if len(self.comment):
            s += f" # {self.comment}"
        return s

    __repr__ = __str__
