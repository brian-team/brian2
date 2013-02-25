'''
Module implementing Brian's refractory mechanism.
'''

from brian2.units.fundamentalunits import Unit
from brian2.units.allunits import second

from .equations import (Equations, SingleEquation, DIFFERENTIAL_EQUATION,
                        PARAMETER, Expression)

__all__ = ['add_refractoriness']

def check_identifier_refractory(identifier):
    '''
    Check that the identifier is not using a name reserved for the refractory
    mechanism. The reserved names are `is_active`, `refractory`,
    `refractory_until`.
    
    Parameters
    ----------
    identifier : str
        The identifier to check.
        
    Raises
    ------
    ValueError
        If the identifier is a variable name used for the refractory mechanism.
    '''
    if identifier in ('is_active', 'refractory', 'refractory_until'):
        raise ValueError(('The name "%s" is used in the refractory mechanism '
                         ' and should not be used as a variable name.' % identifier))

Equations.register_identifier_check(check_identifier_refractory)


def add_refractoriness(eqs):
    '''
    Extends a given set of equations with the refractory mechanism. New
    parameters are added and differential equations with the "active" flag
    are changed so that their right-hand side is 0 when the neuron is
    refractory (by multiplication with the ``is_active`` variable).
    
    Parameters
    ----------
    eqs : `Equations`
        The equations without refractory mechanism.
    
    Returns
    -------
    new_eqs : `Equations`
        New equations, with added parameters and changed differential 
        equations having the "active" flag.    
    '''
    new_equations = []
    
    # replace differential equations having the active flag    
    for eq in eqs.itervalues():
        if eq.eq_type == DIFFERENTIAL_EQUATION and 'active' in eq.flags:
            # the only case where we have to change anything
            new_code = 'is_active*(' + eq.expr.code + ')'
            new_equations.append(SingleEquation(DIFFERENTIAL_EQUATION,
                                                eq.varname, eq.unit,
                                                Expression(new_code)))
        else:
            new_equations.append(eq)
    
    # add new parameters
    new_equations.append(SingleEquation(PARAMETER, 'is_active', Unit(1)))
    new_equations.append(SingleEquation(PARAMETER, 'refractory', second))
    new_equations.append(SingleEquation(PARAMETER, 'refractory_until', second))

    return Equations(new_equations)
