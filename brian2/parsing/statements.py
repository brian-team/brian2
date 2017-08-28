from pyparsing import (CharsNotIn, Optional, Suppress, Word, Regex,
                       ParseException, alphas, nums)

VARIABLE = Word(alphas + '_',
                  alphas + nums + '_').setResultsName('variable')

OP = Regex(r'(\+|\-|\*|/|//|%|\*\*|>>|<<|&|\^|\|)?=').setResultsName('operation')
EXPR = CharsNotIn('#').setResultsName('expression')
COMMENT = CharsNotIn('#').setResultsName('comment')
STATEMENT = VARIABLE + OP + EXPR + Optional(Suppress('#') + COMMENT)

def parse_statement(code):
    '''
    Parses a single line of code into "var op expr".
    
    Parameters
    ----------
    code : str
        A string containing a single statement of the form
        ``var op expr # comment``, where the ``# comment`` part is optional.
    
    Returns
    -------
    var, op, expr, comment : str, str, str, str
        The four parts of the statement.
        
    Examples
    --------
    >>> parse_statement('v = -65*mV  # reset the membrane potential')
    ('v', '=', '-65*mV', 'reset the membrane potential')
    >>> parse_statement('v += dt*(-v/tau)')
    ('v', '+=', 'dt*(-v/tau)', '')
    '''

    try:
        parsed = STATEMENT.parseString(code, parseAll=True)
    except ParseException as p_exc:
        raise ValueError('Parsing the statement failed: \n' + str(p_exc.line) +
                         '\n' + ' ' * (p_exc.column - 1) + '^\n' + str(p_exc))
    if len(parsed['expression'].strip()) == 0:
        raise ValueError(('Empty expression in the RHS of the statement:'
                          '"%s" ') % code)
    return (parsed['variable'].strip(),
            parsed['operation'],
            parsed['expression'].strip(),
            parsed.get('comment', '').strip())
