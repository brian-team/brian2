import operator

from sympy import Wild, Symbol, sympify
import sympy as sp

from brian2.equations import Equations

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



if __name__ == '__main__':
    from brian2 import sin  
    eqs = Equations('''
    dv/dt = sin(2*3.141*t*freq+phase) / tau : 1
    tau : second
    freq : Hz 
    phase : 1
    ''' )
    
    variables, matrix, b = get_linear_system(eqs)
    print 'variables: ', variables
    print '\nCoefficient matrix:'
    print matrix
    print '\nVector b:'
    print b
    print '\n diagonalized:'
    t = Symbol('t')    
    A = (matrix * t).exp()
    print 'A = :'
    print A
    C = sp.Matrix([A.dot(b)]) + b
    print 'C = :'
    print C
#    x(t) = dot(A(t), S(0)) + C(t)
#    x(t + dt) = dot(A(t + dt), S(0)) + C(t + dt)
#    = dot(A(t), S(0)) + dot(A(dt), S(0)) + C(t) + C(dt)
#    = S(t) + dot(A(dt), S(0)) + C(dt)
#    
#    C(0) = 0
        