import re

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
