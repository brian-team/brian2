'''
Utility functions for parsing expressions and statements.
'''
import sympy
from sympy.printing.str import StrPrinter

from brian2.core.functions import DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS, log10
from brian2.parsing.rendering import SympyNodeRenderer


def str_to_sympy(expr):
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
        The string expression to parse..
    
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

    def _print_Function(self, expr):
        # Special workaround for the int function
        if expr.func.__name__ == 'int_':
            return "int(%s)" % self.stringify(expr.args, ", ")
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
