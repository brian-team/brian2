'''
Utility functions for parsing expressions and statements.
'''
import re
from StringIO import StringIO

import sympy
from sympy.parsing.sympy_tokenize import (generate_tokens, untokenize, NUMBER,
                                          NAME, OP)
from .functions.numpyfunctions import DEFAULT_FUNCTIONS

SYMPY_DICT = {'I': sympy.I,
              'Float': sympy.Float,
              'Integer': sympy.Integer,
              'Symbol': sympy.Symbol}

def parse_statement(code):
    '''
    Parses a single line of code into "var op expr".
    
    Parameters
    ----------
    code : str
        A string containing a single statement of the form ``var op expr``.
    
    Returns
    -------
    var, op, expr : str, str, str
        The three parts of the statement.
        
    Examples
    --------
    >>> parse_statement('v = -65*mV')
    ('v', '=', '-65*mV')
    >>> parse_statement('v += dt*(-v/tau)')
    ('v', '+=', 'dt*(-v/tau)')
    
    '''
    m = re.search(r'(\+|\-|\*|/|//|%|\*\*|>>|<<|&|\^|\|)?=', code)
    if not m:
        raise ValueError("Could not extract statement from: " + code)
    start, end = m.start(), m.end()
    op = code[start:end].strip()
    var = code[:start].strip()
    expr = code[end:].strip()
    # var should be a single word
    if len(re.findall(r'^[A-Za-z_][A-Za-z0-9_]*$', var))!=1:
        raise ValueError("LHS in statement must be single variable name, line: " + code)
    
    return var, op, expr

def parse_to_sympy(expr, local_dict=None):
    '''
    Parses a string into a sympy expression. The reason for not using `sympify`
    directly is that sympify does a ``from sympy import *``, adding all functions
    to its namespace. This leads to issues when trying to use sympy function
    names as variable names. For example, both ``beta`` and ``factor`` -- quite
    reasonable names for variables -- are sympy functions, using them as
    variables would lead to a parsing error.
    
    Parameters
    ----------
    expr : str
        The string expression to parse.
    local_dict : dict
        A dictionary mapping names to objects. These names will be left
        untouched and not wrapped in Symbol(...).
    
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
    This function is basically a stripped down version of sympy's
    `~sympy.parsing.sympy_parser._transform` function.  Sympy is licensed
    using the new BSD license:
    https://github.com/sympy/sympy/blob/master/LICENSE
    '''
    if local_dict is None:
        # use the standard functions
        local_dict = dict((name, f.sympy_func) for
                         name, f in DEFAULT_FUNCTIONS.iteritems()
                         if f.sympy_func is not None)
    
    tokens = generate_tokens(StringIO(expr).readline)
    
    result = []
    for toknum, tokval, _, _, _ in tokens:
        
        if toknum == NUMBER:
            postfix = []
            number = tokval
            
            # complex numbers
            if number.endswith('j') or number.endswith('J'):
                number = number[:-1]
                postfix = [(OP, '*'), (NAME, 'I')]

            # floating point numbers
            if '.' in number or (('e' in number or 'E' in number) and
                    not (number.startswith('0x') or number.startswith('0X'))):
                seq = [(NAME, 'Float'), (OP, '('), (NUMBER, repr(str(number))), (OP, ')')]
            # integers
            else:                
                seq = [(NAME, 'Integer'), (OP, '('), (NUMBER, number), (OP, ')')]

            result.extend(seq + postfix)
        elif toknum == NAME:
            name = tokval

            if name in ['True', 'False', 'None'] or name in local_dict:
                result.append((NAME, name))
                continue

            result.extend([
                # TODO: We always assume that variables are real, right?           
                (NAME, 'Symbol'),
                (OP, '('),
                (NAME, repr(str(name))),
                (OP, ','),
                (NAME, 'real'),
                (OP, '='),
                (NAME, 'True'),
                (OP, ')'),
            ])
        elif toknum == OP:
            op = tokval

            if op == '^':
                result.append((OP, '**'))            
            else:
                result.append((OP, op))
        else:
            result.append((toknum, tokval))

    code = untokenize(result)
    
    try:
        s_expr = eval(code, SYMPY_DICT, local_dict)
    except Exception as ex:
        raise ValueError('Expression "%s" could not be parsed: %s' %
                         (expr, str(ex)))

    return s_expr


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

    sympy_expr = sympy_expr.subs(replacements)
    
    return str(sympy_expr)

    