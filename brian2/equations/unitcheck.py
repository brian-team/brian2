'''
Utility functions for handling the units in `Equations`.
'''
import re

from brian2.units.fundamentalunits import (get_unit, Unit,
                                           fail_for_dimension_mismatch,
                                           get_dimensions)

from brian2.parsing.expressions import parse_expression_dimensions
from brian2.parsing.statements import parse_statement
from brian2.core.variables import Variable

__all__ = ['unit_from_expression', 'check_dimensions',
           'check_units_statements']


def check_dimensions(expression, dimensions, variables):
    '''
    Compares the physical dimensions of an expression to expected dimensions in
    a given namespace.

    Parameters
    ----------
    expression : str
        The expression to evaluate.
    dimensions : `Dimension`
        The expected physical dimensions for the `expression`.
    variables : dict
        Dictionary of all variables (including external constants) used in
        the `expression`.

    Raises
    ------
    KeyError
        In case on of the identifiers cannot be resolved.
    DimensionMismatchError
        If an unit mismatch occurs during the evaluation.
    '''
    expr_dims = parse_expression_dimensions(expression, variables)
    err_msg = ('Expression {expr} does not have the '
               'expected unit {expected}').format(expr=expression.strip(),
                                                  expected=repr(get_unit(dimensions)))
    fail_for_dimension_mismatch(expr_dims, dimensions, err_msg)


def check_units_statements(code, variables):
    '''
    Check the units for a series of statements. Setting a model variable has to
    use the correct unit. For newly introduced temporary variables, the unit
    is determined and used to check the following statements to ensure
    consistency.
    
    Parameters
    ----------
    code : str
        The statements as a (multi-line) string
    variables : dict of `Variable` objects
        The information about all variables used in `code` (including
        `Constant` objects for external variables)
    
    Raises
    ------
    KeyError
        In case on of the identifiers cannot be resolved.
    DimensionMismatchError
        If an unit mismatch occurs during the evaluation.
    '''
    variables = dict(variables)
    # Avoid a circular import
    from brian2.codegen.translation import analyse_identifiers
    newly_defined, _, unknown = analyse_identifiers(code, variables)
    
    if len(unknown):
        raise AssertionError(('Encountered unknown identifiers, this should '
                             'not happen at this stage. Unknown identifiers: %s'
                             % unknown))

    
    code = re.split(r'[;\n]', code)
    for line in code:
        line = line.strip()
        if not len(line):
            continue  # skip empty lines
        
        varname, op, expr, comment = parse_statement(line)
        if op in ('+=', '-=', '*=', '/=', '%='):
            # Replace statements such as "w *=2" by "w = w * 2"
            expr = '{var} {op_first} {expr}'.format(var=varname,
                                                    op_first=op[0],
                                                    expr=expr)
            op = '='
        elif op == '=':
            pass
        else:
            raise AssertionError('Unknown operator "%s"' % op) 

        expr_unit = parse_expression_dimensions(expr, variables)

        if varname in variables:
            expected_unit = variables[varname].dim
            fail_for_dimension_mismatch(expr_unit, expected_unit,
                                        ('The right-hand-side of code '
                                         'statement ""%s" does not have the '
                                         'expected unit %r') % (line,
                                                               expected_unit))
        elif varname in newly_defined:
            # note the unit for later
            variables[varname] = Variable(name=varname,
                                          dimensions=get_dimensions(expr_unit),
                                          scalar=False)
        else:
            raise AssertionError(('Variable "%s" is neither in the variables '
                                  'dictionary nor in the list of undefined '
                                  'variables.' % varname))
