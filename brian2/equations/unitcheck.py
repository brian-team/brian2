'''
Utility functions for handling the units in `Equations`.
'''

from brian2.units.fundamentalunits import Quantity, Unit
from brian2.units.fundamentalunits import DIMENSIONLESS
from brian2.units.allunits import (metre, meter, second, amp, kelvin, mole,
                                   candle, kilogram, radian, steradian, hertz,
                                   newton, pascal, joule, watt, coulomb, volt,
                                   farad, ohm, siemens, weber, tesla, henry,
                                   celsius, lumen, lux, becquerel, gray,
                                   sievert, katal, kgram, kgramme)

__all__ = ['get_unit_from_string']


def get_unit_from_string(unit_string):
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
