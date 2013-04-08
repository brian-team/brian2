import operator

from sympy import Wild, Symbol, sympify
import sympy as sp

from brian2.equations import Equations
from brian2.stateupdaters.base import StateUpdateMethod

___all__ = ['linear']

def get_linear_system(eqs):
    diff_eqs = eqs.substituted_expressions
    diff_eq_names = eqs.diff_eq_names
    
    symbols = [Symbol(name) for name in diff_eq_names]
    # Coefficients
    wildcards = [Wild('c_' + name, exclude=symbols) for name in diff_eq_names]
    
    #Additive constant
    constant_wildcard = Wild('c', exclude=symbols)
    
    pattern = reduce(operator.add, [c * s for c, s in zip(wildcards, symbols)])
    pattern += constant_wildcard
    
    coefficients = sp.zeros(len(diff_eq_names))
    constants = sp.zeros((len(diff_eq_names), 1))
    
    for row_idx, (name, expr) in enumerate(diff_eqs):
        s_expr = sympify(expr, locals=dict([(s.name, s) for s in symbols])).expand()
        #print s_expr.expand()
        pattern_matches = s_expr.match(pattern)
        if pattern_matches is None:
            raise ValueError(('The expression "%s", defining the variable %s, '
                             'could not be separated into linear components') %
                             (str(s_expr), name))
        
        for col_idx in xrange(len(diff_eq_names)):
            coefficients[row_idx, col_idx] = pattern_matches[wildcards[col_idx]]
        
        constants[row_idx] = pattern_matches[constant_wildcard]

    return (diff_eq_names, coefficients, constants)


class LinearStateUpdater(StateUpdateMethod):    
    
    def can_integrate(self, equations, namespace, specifiers):
        return False
    
    def __call__(self, equations):
        # Get a representation of the ODE system in the form of
        # dX/dt = M*X + B
        variables, matrix, constants = get_linear_system(eqs)
        
        
        symbols = [Symbol(variable) for variable in variables]
        solution = sp.solve_linear_system(matrix.row_join(constants), *symbols)
        b = sp.Matrix([solution[symbol] for symbol in symbols]).transpose()
        
        # Solve the system
        dt = Symbol('dt')    
        A = (matrix * dt).exp()                
        C = -sp.Matrix([A.dot(b)]) + b
        S = sp.MatrixSymbol('_S', len(variables), 1)
        updates = A * S + C.transpose()
        
        # The solution contains _S_00, _S_10 etc. for the state variables,
        # replace them with the state variable names 
        abstract_code = []
        for idx, (variable, update) in enumerate(zip(variables, updates)):
            abstract_code.append(variable + ' = ' +
                                 str(update.subs('_S_%d0' % idx, variable)))
        
        return '\n'.join(abstract_code)

linear = LinearStateUpdater()

# The linear state updater has the highest priority
StateUpdateMethod.register('linear', linear, 0)

if __name__ == '__main__':
    eqs = Equations('''
    dv/dt = -v / tau + const_v: 1
    du/dt = -u / tau + const_u: 1
    tau : second
    const_v : 1
    const_u : 1
    ''' )
    
    print 'Equations:'
    print eqs
    print 'Abstract code'
    print linear(eqs)