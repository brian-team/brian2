'''
AST parsing based analysis of expressions
'''

import ast

from brian2.units.fundamentalunits import (get_unit_fast,
                                           DimensionMismatchError,
                                           have_same_dimensions,
                                           )
from brian2.units import allunits
from brian2.units import stdunits

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


standard_unit_map = dict()
for k in allunits.__all__:
    standard_unit_map[k] = get_unit_fast(getattr(allunits, k))
for k in stdunits.__all__:
    standard_unit_map[k] = get_unit_fast(getattr(stdunits, k))    
    
    
def parse_expression_unit(expr, varunits, funcunits, use_standard_units=True):
    '''
    Returns the unit value of an expression, and checks its validity
    
    Parameters
    ----------
    expr : str
        The expression to check.
    varunits : dict
        A mapping of (name, unit) pairs.
    funcunits : dict
        TODO: support for functions.
    use_standard_units : bool
        Whether or not to include the allunits and stdunits units.
    
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
    if isinstance(expr, str):
        mod = ast.parse(expr, mode='eval')
        expr = mod.body
    # add standard unit names
    unitmap = {'True': get_unit_fast(1),
               'False': get_unit_fast(1),
               }
    if use_standard_units:
        unitmap.update(standard_unit_map)
    unitmap.update(varunits)
    
    if expr.__class__ is ast.Name:
        return unitmap[expr.id]
    elif expr.__class__ is ast.Num:
        return get_unit_fast(1)
    elif expr.__class__ is ast.BoolOp:
        # check that the units are valid in each subexpression
        for node in expr.values:
            parse_expression_unit(node, unitmap, funcunits, use_standard_units=False)
        # but the result is a bool, so we just return 1 as the unit
        return get_unit_fast(1)
    elif expr.__class__ is ast.Compare:
        # check that the units are consistent in each subexpression
        subexprs = [expr.left]+expr.comparators
        subunits = []
        for node in subexprs:
            subunits.append(parse_expression_unit(node, unitmap, funcunits, use_standard_units=False))
        for left, right in zip(subunits[:-1], subunits[1:]):
            if not have_same_dimensions(left, right):
                raise DimensionMismatchError("Comparison of expressions with different units",
                                             *[u.dim for u in subunits])
        # but the result is a bool, so we just return 1 as the unit
        return get_unit_fast(1)
    elif expr.__class__ is ast.Call:
        # We will later on have a way to specify units for functions, but
        # it depends on the data structure. The steps to complete this are:
        # 1. Check all the function arguments for unit consistency and 
        #    get their units. Raise an error if the user uses *args, **kwds
        #    or keywords, as we only allow positional arguments for now.
        #    See NodeRenderer.render_Call for more details on this.
        # 2. With this, we check the function arguments against the function
        #    signature provided by the user and make sure it is OK. The
        #    function signature can restrict the values it is passed, and
        #    either provide a single output unit, or an output unit that is
        #    a function of its input units.
        raise NotImplementedError
    elif expr.__class__ is ast.BinOp:
        op = expr.op.__class__.__name__
        left = parse_expression_unit(expr.left, unitmap, funcunits, use_standard_units=False)
        right = parse_expression_unit(expr.right, unitmap, funcunits, use_standard_units=False)
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
        u = parse_expression_unit(expr.operand, unitmap, funcunits, use_standard_units=False)
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
