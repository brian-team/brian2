import sympy as sp

from brian2.stateupdaters.base import StateUpdateMethod

__all__ = ['exponential_euler']

def get_conditionally_linear_system(eqs):
    '''
    Convert equations into a linear system using sympy.
    
    Parameters
    ----------
    eqs : `Equations`
        The model equations.
    
    Returns
    -------
    
    
    Raises
    ------
    ValueError
        If the equations cannot be converted into an M * X + B form.
    '''
    diff_eqs = eqs.substituted_expressions
    
    coefficients = {}
    
    for name, expr in diff_eqs:
        var = sp.Symbol(name, real=True)        
        
        # Coefficients
        wildcard = sp.Wild('_A', exclude=[var])
        #Additive constant
        constant_wildcard = sp.Wild('_B', exclude=[var])    
        pattern = wildcard*var + constant_wildcard
    
        # Factor out the variable
        s_expr = sp.collect(expr.sympy_expr.expand(), var)
        matches = s_expr.match(pattern)
        
        if matches is None:
            raise ValueError(('The expression "%s", defining the variable %s, '
                             'could not be separated into linear components') %
                             (s_expr, name))
        coefficients[name] = (matches[wildcard], matches[constant_wildcard])
    
    return coefficients

class ExponentialEulerStateUpdater(StateUpdateMethod):
    def can_integrate(self, equations, namespace, specifiers):
        if equations.is_stochastic:
            return False
        
        # Try whether the equations are conditionally linear
        try:
            _ = get_conditionally_linear_system(equations)
        except ValueError:
            return False
        
        return True
    
    def __call__(self, equations, namespace=None, specifiers=None):
        system = get_conditionally_linear_system(equations)
        
        code = []
        for var, (A, B) in system.iteritems():
            A_name = '_A_' + var
            BA_name = '_BA_' + var
            # Calculate A and B/A
            code += [A_name + ' = ' + str(A)]
            code += [BA_name + ' = ' + str(B/A)]
            # The actual update step
            update = '_{var} = ({var} + {BA})*exp(-{A} * dt) - BA'
            code += [update.format(var=var, A=A_name, BA=BA_name)]
        
        # Replace all the variables with their updated value
        for var in system:
            code += ['{var} = _{var}'.format(var=var)]
            
        return '\n'.join(code)

exponential_euler = ExponentialEulerStateUpdater() 