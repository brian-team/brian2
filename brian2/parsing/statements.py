from __future__ import absolute_import
from pyparsing import (CharsNotIn, OneOrMore, Optional, Suppress, Word, Regex,
                       ParseException, alphas, nums)

from brian2.utils.caching import cached

VARIABLE = Word(alphas + '_',
                  alphas + nums + '_').setResultsName('variable', listAllMatches=True)

OP = Regex(r'(\+|\-|\*|/|//|%|\*\*|>>|<<|&|\^|\|)?=').setResultsName('operation')
EXPR = CharsNotIn('#').setResultsName('expression')
COMMENT = CharsNotIn('#').setResultsName('comment')
SINGLE_ASSIGNMENT = VARIABLE + OP + EXPR + Optional(Suppress('#') + COMMENT)
TUPLE_ASSIGNMENT = OneOrMore(VARIABLE + ',') + VARIABLE + OP + EXPR + Optional(Suppress('#') + COMMENT)
STATEMENT = TUPLE_ASSIGNMENT | SINGLE_ASSIGNMENT


@cached
def parse_statement(code):
    '''
    parse_statement(code)

    Parses a single line of code into "vars op expr".
    
    Parameters
    ----------
    code : str
        A string containing a single statement of the form
        ``vars op expr # comment``, where the ``# comment`` part is optional.
    
    Returns
    -------
    vars, op, expr, comment : str, str, str, str
        The four parts of the statement.
        
    Examples
    --------
    >>> parse_statement('v = -65*mV  # reset the membrane potential')
    (('v',), '=', '-65*mV', 'reset the membrane potential')
    >>> parse_statement('v += dt*(-v/tau)')
    (('v',), '+=', 'dt*(-v/tau)', '')
    >>> parse_statement('x, y = my_func(z)')
    (('x', 'y'), '=', 'my_func(z)', '')
    '''
    try:
        parsed = STATEMENT.parseString(code, parseAll=True)
    except ParseException as p_exc:
        raise ValueError('Parsing the statement failed: \n' + str(p_exc.line) +
                         '\n' + ' ' * (p_exc.column - 1) + '^\n' + str(p_exc))
    if len(parsed['expression'].strip()) == 0:
        raise ValueError(('Empty expression in the RHS of the statement:'
                          '"%s" ') % code)
    variable = tuple(var.strip() for var in parsed['variable'])
    parsed_statement = (variable,
                        parsed['operation'],
                        parsed['expression'].strip(),
                        parsed.get('comment', '').strip())

    return parsed_statement
