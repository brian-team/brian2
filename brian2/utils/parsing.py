'''
Utility functions for parsing expressions to sympy.
'''
from StringIO import StringIO

import sympy
from sympy.parsing.sympy_tokenize import (generate_tokens, untokenize, NUMBER,
                                          NAME, OP)

SYMPY_DICT = {'I': sympy.I,
              'Float': sympy.Float,
              'Integer': sympy.Integer,
              'Symbol': sympy.Symbol}

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
        local_dict = {}
    
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
                (NAME, 'Symbol'),
                (OP, '('),
                (NAME, repr(str(name))),
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
        raise SyntaxError('Expression "%s" could not be parsed: %s' %
                          (expr, str(ex)))

    return s_expr