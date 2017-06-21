from brian2.parsing.sympytools import sympy_to_str, str_to_sympy

from .base import StateUpdateMethod, UnsupportedEquationsException

__all__ = ['GSL_stateupdater']

class GSLStateUpdater(StateUpdateMethod): #TODO: specify GSL integrator?
    '''
    A statupdater that rewrites the differential equations so that the GSL templater knows how to write the
    code in the target language.
    '''
    def __call__(self, equations, variables=None):

        if equations.is_stochastic:
            raise UnsupportedEquationsException('Cannot solve stochastic '
                                                'equations with this state '
                                                'updater.')

        # the approach is to 'tag' the differential equation variables so they can
        # be translated to GSL code
        diff_eqs = equations.get_substituted_expressions(variables)

        code = []
        count_statevariables = 0
        counter = {}

        for diff_name, expr in diff_eqs:
            counter[diff_name] = count_statevariables
            code += ['_gsl_{var}_f{count} = {expr}'.format(var=diff_name,
                                                               expr=expr,
                                                               count=counter[diff_name])]
            count_statevariables += 1


        return ('\n').join(code)

    # Copy doc from parent class
    __call__.__doc__ = StateUpdateMethod.__call__.__doc__

GSL_stateupdater = GSLStateUpdater()
