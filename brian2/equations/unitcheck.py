'''
Utility functions for handling the units in `Equations`.
'''

from brian2.units.fundamentalunits import Quantity, Unit,\
    fail_for_dimension_mismatch, DimensionMismatchError
from brian2.units.fundamentalunits import DIMENSIONLESS
from brian2.units.allunits import (metre, meter, second, amp, kelvin, mole,
                                   candle, kilogram, radian, steradian, hertz,
                                   newton, pascal, joule, watt, coulomb, volt,
                                   farad, ohm, siemens, weber, tesla, henry,
                                   celsius, lumen, lux, becquerel, gray,
                                   sievert, katal, kgram, kgramme)

from brian2.codegen.translation import analyse_identifiers
from brian2.utils.stringtools import get_identifiers
from brian2.codegen.parsing import parse_statement
from brian2.core.specifiers import VariableSpecifier

__all__ = ['unit_from_string', 'unit_from_expression', 'check_unit',
           'check_units_statements']


def unit_from_string(unit_string):
    '''
    Returns the unit that results from evaluating a string like
    "siemens / metre ** 2", allowing for the special string "1" to signify
    dimensionless units.
    
    Parameters
    ----------    
    unit_string : str
        The string that should evaluate to a unit
    
    Returns
    -------
    u : Unit
        The resulting unit
    
    Raises
    ------
    ValueError
        If the string cannot be evaluated to a unit.
    '''
    
    # We avoid using DEFAULT_NUMPY_NAMESPACE here as importing core.namespace
    # would introduce a circular dependency between it and the equations
    # package
    base_units = [metre, meter, second, amp, kelvin, mole, candle, kilogram,
                  radian, steradian, hertz, newton, pascal, joule, watt,
                  coulomb, volt, farad, ohm, siemens, weber, tesla, henry,
                  celsius, lumen, lux, becquerel, gray, sievert, katal, kgram,
                  kgramme]
    namespace = dict((repr(unit), unit) for unit in base_units)
    namespace['Hz'] = hertz  # Also allow Hz instead of hertz
    unit_string = unit_string.strip()
    
    # Special case: dimensionless unit
    if unit_string == '1':
        return Unit(1, dim=DIMENSIONLESS)
    
    # Check first whether the expression evaluates at all, using only base units
    try:
        evaluated_unit = eval(unit_string, namespace)
    except Exception as ex:
        raise ValueError(('"%s" does not evaluate to a unit when only using '
                          'base units (e.g. volt but not mV): %s') %
                         (unit_string, ex))
    
    # Check whether the result is a unit
    if not isinstance(evaluated_unit, Unit):
        if isinstance(evaluated_unit, Quantity):
            raise ValueError(('"%s" does not evaluate to a unit but to a '
                              'quantity -- make sure to only use units, e.g. '
                              '"siemens/metre**2" and not "1 * siemens/metre**2"') %
                             unit_string)
        else:
            raise ValueError(('"%s" does not evaluate to a unit, the result '
                             'has type %s instead.' % (unit_string,
                                                       type(evaluated_unit))))

    # No error has been raised, all good
    return evaluated_unit


def unit_from_expression(expression, namespace, specifiers):
    '''
    Evaluates the unit for an expression in a given namespace.
    
    Parameters
    ----------
    expression : str or `Expression`
        The expression to evaluate.
    namespace : dict-like
        The namespace of external variables.
    specifiers : dict of `Specifier` objects
        The information about the internal variables
    
    Returns
    -------
    q : Quantity
        The quantity or unit for the expression
    
    Raises
    ------
    KeyError
        In case on of the identifiers cannot be resolved.
    DimensionMismatchError
        If an unit mismatch occurs during the evaluation.    
    '''
    
    # Make sure we have a string expression
    expression = str(expression)
    
    # Create a mapping with all identifier names to either their actual
    # value (for external identifiers) or their unit (for specifiers)
    unit_namespace = {}
    identifiers = get_identifiers(expression)
    for name in identifiers:
        if name in specifiers:
            unit_namespace[name] = specifiers[name].unit
        else:
            # TODO: Should we add special support for user-defined functions
            #       or should we just assume that the Python version takes
            #       care of units?
            # This raises an error if the identifier cannot be resolved
            unit_namespace[name] = namespace[name]
    
    try:
        unit = eval(expression, unit_namespace)
    except DimensionMismatchError as dim_ex:
        raise DimensionMismatchError(('A unit mismatch occured while '
                                      'evaluating the expression "%s": %s ' %
                                      (expression, dim_ex.desc)),
                                     *dim_ex.dims)
    
    if not isinstance(unit, Quantity):
        # It might be a dimensionless array
        unit = Quantity(unit)
    
    return unit


def check_unit(expression, unit, namespace, specifiers):
    '''
    Evaluates the unit for an expression in a given namespace.
    
    Parameters
    ----------
    expression : str
        The expression to evaluate.
    namespace : dict-like
        The namespace of external variables.
    specifiers : dict of `Specifier` objects
        The information about the internal variables
    
    Raises
    ------
    KeyError
        In case on of the identifiers cannot be resolved.
    DimensionMismatchError
        If an unit mismatch occurs during the evaluation.
    
    See Also
    --------
    unit_from_expression
    '''
    expr_unit = unit_from_expression(expression, namespace, specifiers)
    
    fail_for_dimension_mismatch(expr_unit, unit, ('Expression %s does not '
                                                  'have the expected units' %
                                                  expression))

def check_units_statements(code, namespace, specifiers):
    '''
    Check the units for a series of statements. Setting a model variable has to
    use the correct unit. For newly introduced temporary variables, the unit
    is determined and used to check the following statements to ensure
    consistency.
    
    Parameters
    ----------
    expression : str
        The expression to evaluate.
    namespace : dict-like
        The namespace of external variables.
    specifiers : dict of `Specifier` objects
        The information about the internal variables
    
    Raises
    ------
    KeyError
        In case on of the identifiers cannot be resolved.
    DimensionMismatchError
        If an unit mismatch occurs during the evaluation.
    '''
    known = set(specifiers.keys()) | set(namespace.keys())
    newly_defined, _, unknown = analyse_identifiers(code, known)
    
    if len(unknown):
        raise AssertionError(('Encountered unknown identifiers, this should '
                             'not happen at this stage. Unkown identifiers: %s'
                             % unknown))
    
    # We want to add newly defined variables to the specifiers dictionary so we
    # make a copy now
    specs = dict(specifiers)
    
    code = code.split('\n')
    for line in code:
        line = line.strip()
        if not len(line):
            continue  # skip empty lines
        
        var, op, expr = parse_statement(line)
        if op in ('+=', '-=', '*=', '/=', '%='):
            # Replace statements such as "w *=2" by "w = w * 2"
            expr = '{var} {op_first} {expr}'.format(var=var,
                                                    op_first=op[0],
                                                    expr=expr)
            op = '='
        elif op == '=':
            pass
        else:
            raise AssertionError('Unknown operator "%s"' % op) 
    
        expr_unit = unit_from_expression(expr, namespace, specs)
        
        if var in specifiers:
            fail_for_dimension_mismatch(specifiers[var].unit,
                                        expr_unit,
                                        ('Code statement "%s" does not use '
                                         'correct units' % line))
        elif var in newly_defined:
            # note the unit for later
            specs[var] = VariableSpecifier(var, expr_unit)
        else:
            raise AssertionError(('Variable "%s" is neither in the specifiers '
                                  'dictionary nor in the list of undefined '
                                  'variables.' % var))