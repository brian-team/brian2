'''
AST parsing based analysis of expressions
'''

import ast

from brian2.units.fundamentalunits import (Unit, get_unit_fast,
                                           DimensionMismatchError,
                                           have_same_dimensions,
                                           )

__all__ = ['is_boolean_expression',
           'parse_expression_unit',]

def is_boolean_expression(expr, boolvars=None, boolfuncs=None):
    '''
    Determines if an expression is of boolean type or not
    
    Parameters
    ----------
    
    expr : str
        The expression to test
    boolvars : set
        The set of variables of boolean type.
    boolfuncs : set
        The set of functions which return booleans.

    Returns
    -------
    
    isbool : bool
        Whether or not the expression is boolean.

    Raises
    ------
    
    SyntaxError
        If the expression ought to be boolean but is not,
        for example ``x<y and z`` where ``z`` is not a boolean variable.
        
    Details
    -------
    
    We test the following cases recursively on the abstract syntax tree:
    
    * The node is a boolean operation. If all the subnodes are boolean
      expressions we return ``True``, otherwise we raise the ``SyntaxError``.
    * The node is a function call, we return ``True`` or ``False`` depending
      on whether the name is in ``boolfuncs``.
    * The node is a variable name, we return ``True`` or ``False`` depending
      on whether the name is in ``boolvars`` or if the name is ``True`` or
      ``False``.
    * The node is a comparison, we return ``True``.
    * The node is a unary operation, we return ``True`` if the operation is
      ``not``, otherwise ``False``.
    * Otherwise we return ``False``.
    '''
    if boolfuncs is None:
        boolfuncs = set([])
    if boolvars is None:
        boolvars = set([])
        
    boolvars.add('True')
    boolvars.add('False')
    
    # If we are working on a string, convert to the top level node    
    if isinstance(expr, str):
        mod = ast.parse(expr, mode='eval')
        expr = mod.body
        
    if expr.__class__ is ast.BoolOp:
        if all(is_boolean_expression(node, boolvars, boolfuncs) for node in expr.values):
            return True
        else:
            raise SyntaxError("Expression ought to be boolean but is not (e.g. 'x<y and 3')")
    elif expr.__class__ is ast.Name:
        return expr.id in boolvars
    elif expr.__class__ is ast.Call:
        return expr.func.id in boolfuncs
    elif expr.__class__ is ast.Compare:
        return True
    elif expr.__class__ is ast.UnaryOp:
        return expr.op.__class__.__name__ == 'Not'
    else:
        return False
    
    
def parse_expression_unit(expr, namespace, specifiers):
    '''
    Returns the unit value of an expression, and checks its validity
    
    Parameters
    ----------
    expr : str
        The expression to check.
    namespace : dict-like
        The namespace of external variables.
    specifiers : dict of `Specifier` objects
        The information about the internal variables
    
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
        
    Notes
    -----
    
    Currently, functions do not work, see comments in function.
    '''
    # If we are working on a string, convert to the top level node    
    if isinstance(expr, basestring):
        mod = ast.parse(expr, mode='eval')
        expr = mod.body
    
    if expr.__class__ is ast.Name:
        name = expr.id
        if name in specifiers:
            return specifiers[name].unit
        elif name in namespace:
            return get_unit_fast(namespace[name])
        elif name in ['True', 'False']:
            return Unit(1)
        else:
            raise ValueError('Unknown identifier %s' % name)
    elif expr.__class__ is ast.Num:
        return get_unit_fast(1)
    elif expr.__class__ is ast.BoolOp:
        # check that the units are valid in each subexpression
        for node in expr.values:
            parse_expression_unit(node, namespace, specifiers)
        # but the result is a bool, so we just return 1 as the unit
        return get_unit_fast(1)
    elif expr.__class__ is ast.Compare:
        # check that the units are consistent in each subexpression
        subexprs = [expr.left]+expr.comparators
        subunits = []
        for node in subexprs:
            subunits.append(parse_expression_unit(node, namespace, specifiers))
        for left, right in zip(subunits[:-1], subunits[1:]):
            if not have_same_dimensions(left, right):
                raise DimensionMismatchError("Comparison of expressions with different units",
                                             *[u.dim for u in subunits])
        # but the result is a bool, so we just return 1 as the unit
        return get_unit_fast(1)
    elif expr.__class__ is ast.Call:
        if len(expr.keywords):
            raise ValueError("Keyword arguments not supported.")
        elif expr.starargs is not None:
            raise ValueError("Variable number of arguments not supported")
        elif expr.kwargs is not None:
            raise ValueError("Keyword arguments not supported")

        arg_units = [parse_expression_unit(arg, namespace, specifiers)
                     for arg in expr.args]

        func = namespace.get(expr.func.id, specifiers.get(expr.func, None))
        if func is None:
            raise SyntaxError('Unknown function %s' % expr.func.id)
        if not hasattr(func, '_arg_units') or not hasattr(func, '_return_unit'):
            raise ValueError(('Function %s does not specify how it '
                              'deals with units.') % expr.func.id)

        for idx, arg_unit in enumerate(arg_units):
            # A "None" in func._arg_units means: No matter what unit
            if (func._arg_units[idx] is not None and
                    not have_same_dimensions(arg_unit, func._arg_units[idx])):
                raise DimensionMismatchError(('Argument number %d for function '
                                              '%s does not have the correct '
                                              'units' % (idx + 1, expr.func.id)),
                                             arg_unit, func._arg_units[idx])

        if isinstance(func._return_unit, (Unit, int)):
            # Function always returns the same unit
            return get_unit_fast(func._return_unit)
        else:
            # Function returns a unit that depends on the arguments
            return func._return_unit(*arg_units)

    elif expr.__class__ is ast.BinOp:
        op = expr.op.__class__.__name__
        left = parse_expression_unit(expr.left, namespace, specifiers)
        right = parse_expression_unit(expr.right, namespace, specifiers)
        if op=='Add' or op=='Sub':
            u = left+right
        elif op=='Mult':
            u = left*right
        elif op=='Div':
            u = left/right
        elif op=='Pow':
            if have_same_dimensions(left, 1) and have_same_dimensions(right, 1):
                return get_unit_fast(1)
            if expr.right.__class__ is not ast.Num:
                raise SyntaxError("Cannot parse unit expression with variable power")
            u = left**expr.right.n
        elif op=='Mod':
            u = left % right
        else:
            raise SyntaxError("Unsupported operation "+op)
        return get_unit_fast(u)
    elif expr.__class__ is ast.UnaryOp:
        op = expr.op.__class__.__name__
        # check validity of operand and get its unit
        u = parse_expression_unit(expr.operand, namespace, specifiers)
        if op=='Not':
            return get_unit_fast(1)
        else:
            return u


if __name__=='__main__':
    if 1:
        from brian2.units import volt, amp
        print parse_expression_unit('a%(b*c)', {'a':volt*amp, 'b':volt, 'c':amp}, {})
    if 0:
        EVF = [
            (True, 'a or b', ['a', 'b'], []),
            (True, 'True', [], []),
            (True, 'a<b', [], []),
            (False, 'a+b', [], []),
            (True, 'f(x)', [], ['f']),
            (False, 'f(x)', [], []),
            (True, 'f(x) or a<b and c', ['c'], ['f']),
            ]
        for expect, expr, vars, funcs in EVF:
            print '%s %s "%s" %s %s' % (expect,
                                        is_boolean_expression(expr, set(vars), set(funcs)),
                                        expr, vars, funcs)
        try:
            is_boolean_expression('x<y and z')
        except SyntaxError as e:
            print '"x<y and z" raises', e
        
        try:
            is_boolean_expression('a or b')
        except SyntaxError as e:
            print '"a or b" raises', e
