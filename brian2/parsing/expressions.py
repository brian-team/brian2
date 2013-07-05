import ast

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
        return expr.op.__class__.__name__ is 'not'
    else:
        return False


if __name__=='__main__':
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
