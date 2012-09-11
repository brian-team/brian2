from brian2.units.stdunits import stdunits
from brian2.units.fundamentalunits import (Quantity, Unit, all_registered_units,
                                           DIMENSIONLESS)

__all__ = ['get_default_unit_namespace', 'get_unit_from_string']


def get_default_unit_namespace():    
    namespace = dict([(u.name, u) for u in all_registered_units()])
    namespace.update(stdunits)
    return namespace


def get_unit_from_string(unit_string, unit_namespace=None,
                         only_base_units=False):
    '''
    Returns the unit that results from evaluating a string like
    "siemens / cm ** 2", allowing for the special string "1" to signify
    dimensionless units.
    
    Raises ValueErrors if it is not possible to evaluate the string to a unit.
    
    Arguments:
    
    ``unit_string``
        The string that should evaluate to a unit
    
    ``unit_namespace``
        An optional namespace containing units. If not given, defaults to all
        the units returned by :meth:`get_default_unit_namespace`.
    
    ``only_base_units``
        Whether to allow only units evaluating to 1, e.g. "metre" but not "cm".
        Defaults to ``False``.
    '''
    
    if unit_namespace is None:
        namespace = get_default_unit_namespace()
    else:
        namespace = unit_namespace
    unit_string = unit_string.strip()
    
    # Special case: dimensionless unit
    if unit_string == '1':
        return Unit(1, dim=DIMENSIONLESS)
    
    # Check first whether the expression evaluates at all, using only
    # registered units
    try:
        evaluated_unit = eval(unit_string, namespace)
    except Exception as ex:
        raise ValueError('"%s" does not evaluate to a unit: %s' %
                         (unit_string, ex))
    
    # Check whether the result is a unit
    if not isinstance(evaluated_unit, Unit):
        if isinstance(evaluated_unit, Quantity):
            raise ValueError(('"%s" does not evaluate to a unit but to a '
                              'quantity -- make sure to only use units, e.g. '
                              '"siemens/m**2" and not "1 * siemens/m**2"') %
                             unit_string)
        else:
            raise ValueError(('"%s" does not evaluate to a unit, the result '
                             'has type %s instead.' % (unit_string,
                                                       type(evaluated_unit))))
    # We only want base units, otherwise e.g. setting a unit to mV might lead to 
    # unexpected results (as it is internally saved in volts)
    # TODO: Maybe this restriction is unnecessary with unit arrays?
    if only_base_units and float(evaluated_unit) != 1.0:
        raise ValueError(('"%s" is not a base unit, but only base units are '
                         'allowed in the units part of equations.') % unit_string)

    # No error has been raised, all good
    return evaluated_unit
