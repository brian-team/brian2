'''
Utility functions for parsing expressions and statements.
'''
import re
from collections import Counter

import sympy
from sympy.printing.precedence import precedence
from sympy.printing.str import StrPrinter

from brian2.core.functions import (DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS, log10,
                                   Function)
from brian2.parsing.rendering import SympyNodeRenderer


def check_expression_for_multiple_stateful_functions(expr, variables):
    identifiers = re.findall('\w+', expr)
    identifier_count = Counter(identifiers)
    for identifier, count in identifier_count.iteritems():
        if isinstance(variables.get(identifier, None), Function):
            if not variables[identifier].stateless and count > 1:
                raise NotImplementedError(('The expression "{expr}" contains '
                                           'more than one call of {func}, this '
                                           'is currently not supported since '
                                           '{func} is a stateful function and '
                                           'its multiple calls might be '
                                           'treated incorrectly (e.g.'
                                           '"rand() - rand()" could be '
                                           ' simplified to '
                                           '"0.0").').format(expr=expr,
                                                             func=identifier))


def str_to_sympy(expr, variables=None):
    '''
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
    
    Notes
    -----
    Parsing is done in two steps: First, the expression is parsed and rendered
    as a new string by `SympyNodeRenderer`, translating function names (e.g.
    `ceil` to `ceiling`) and operator names (e.g. `and` to `&`), all unknown
    names are wrapped in `Symbol(...)` or `Function(...)`. The resulting string
    is then evaluated in the `from sympy import *` namespace.
    '''
    if variables is None:
        variables = {}
    check_expression_for_multiple_stateful_functions(expr, variables)
    namespace = {}
    exec 'from sympy import *' in namespace
    # also add the log10 function to the namespace
    namespace['log10'] = log10
    namespace['_vectorisation_idx'] = sympy.Symbol('_vectorisation_idx')
    rendered = SympyNodeRenderer().render_expr(expr)

    try:
        s_expr = eval(rendered, namespace)
    except (TypeError, ValueError, NameError) as ex:
        raise SyntaxError('Error during evaluation of sympy expression: '
                          + str(ex))

    return s_expr


class CustomSympyPrinter(StrPrinter):
    '''
    Printer that overrides the printing of some basic sympy objects. E.g.
    print "a and b" instead of "And(a, b)".
    '''

    def _print_And(self, expr):
        return ' and '.join(['(%s)' % self.doprint(arg) for arg in expr.args])

    def _print_Or(self, expr):
        return ' or '.join(['(%s)' % self.doprint(arg) for arg in expr.args])

    def _print_Not(self, expr):
        if len(expr.args) != 1:
            raise AssertionError('"Not" with %d arguments?' % len(expr.args))
        return 'not (%s)' % self.doprint(expr.args[0])

    def _print_Relational(self, expr):
        return '%s %s %s' % (self.parenthesize(expr.lhs, precedence(expr)),
                             self._relationals.get(expr.rel_op) or expr.rel_op,
                             self.parenthesize(expr.rhs, precedence(expr)))

    def _print_Function(self, expr):
        # Special workaround for the int function
        if expr.func.__name__ == 'int_':
            return "int(%s)" % self.stringify(expr.args, ", ")
        elif expr.func.__name__ == 'Mod':
            return '((%s)%%(%s))' % (self.doprint(expr.args[0]), self.doprint(expr.args[1]))
        else:
            return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")

PRINTER = CustomSympyPrinter()


def sympy_to_str(sympy_expr):
    '''
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
    '''
    
    # replace the standard functions by our names if necessary
    replacements = dict((f.sympy_func, sympy.Function(name)) for
                        name, f in DEFAULT_FUNCTIONS.iteritems()
                        if f.sympy_func is not None and isinstance(f.sympy_func,
                                                                   sympy.FunctionClass)
                        and str(f.sympy_func) != name)
    # replace constants with our names as well
    replacements.update(dict((c.sympy_obj, sympy.Symbol(name)) for
                             name, c in DEFAULT_CONSTANTS.iteritems()
                             if str(c.sympy_obj) != name))

    # Replace _vectorisation_idx by an empty symbol
    replacements[sympy.Symbol('_vectorisation_idx')] = sympy.Symbol('')
    for old, new in replacements.iteritems():
        if sympy_expr.has(old):
            sympy_expr = sympy_expr.subs(old, new)

    return PRINTER.doprint(sympy_expr)


def replace_constants(sympy_expr, variables=None):
    '''
    Replace constant values in a sympy expression with their numerical value.

    Parameters
    ----------
    sympy_expr : `sympy.Expr`
        The expression
    variables : dict-like, optional
        Dictionary of `Variable` objects

    Returns
    -------
    new_expr : `sympy.Expr`
        Expressions with all constants replaced
    '''
    if variables is None:
        return sympy_expr

    symbols = set([symbol for symbol in sympy_expr.atoms()
                   if isinstance(symbol, sympy.Symbol)])
    for symbol in symbols:
        symbol_str = str(symbol)
        if symbol_str in variables:
            var = variables[symbol_str]
            if (getattr(var, 'scalar', False) and
                    getattr(var, 'constant', False)):
                # TODO: We should handle variables of other data types better
                float_val = var.get_value()
                sympy_expr = sympy_expr.xreplace({symbol: sympy.Float(float_val)})

    return sympy_expr


def expression_complexity(expr, complexity=None):
    '''
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
    '''
    if isinstance(expr, str):
        # we do this because sympy.count_ops doesn't handle inequalities (TODO: handle sympy as well str)
        for op in ['<=', '>=', '==', '<', '>']:
            expr = expr.replace(op, '+')
        # work around bug with rand() and randn() (TODO: improve this)
        expr = expr.replace('rand()', 'rand(0)')
        expr = expr.replace('randn()', 'randn(0)')
    subs = {'ADD':1, 'DIV':2, 'MUL':1, 'SUB':1}
    if complexity is not None:
        subs.update(complexity)
    ops = sympy.count_ops(expr, visual=True)
    for atom in ops.atoms():
        if hasattr(atom, 'name'):
            subs[atom.name] = 20 # unknown operations assumed to have a large cost
    return ops.evalf(subs=subs)
