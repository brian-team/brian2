import operator

from sympy import Wild, Symbol, sympify
import sympy as sp

from brian2.core.specifiers import Value
from brian2.utils.stringtools import get_identifiers
from brian2.utils.logger import get_logger
from brian2.stateupdaters.base import StateUpdateMethod

__all__ = ['linear']

logger = get_logger(__name__)

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
        if equations.is_stochastic:
            return False
               
        # Not very efficient but guaranteed to give the correct answer:
        # Just try to apply the integration method
        try:
            self.__call__(equations, namespace, specifiers)
        except (ValueError, NotImplementedError, TypeError) as ex:
            logger.debug('Cannot use linear integration: %s' % ex)
            return False
        
        # It worked
        return True
    
    def __call__(self, equations, namespace=None, specifiers=None):
        
        if namespace is None:
            namespace = {}
        if specifiers is None:
            specifiers = {}
        
        # Get a representation of the ODE system in the form of
        # dX/dt = M*X + B
        variables, matrix, constants = get_linear_system(equations)        
        
        # Make sure that the matrix M is constant, i.e. it only contains
        # external variables or constant specifiers
        # As every symbol in the matrix should be either in the namespace or
        # the specifiers dictionary, it should be sufficient to just check for
        # the presence of any non-constant specifiers.
        for spec in specifiers.itervalues():
            if (any(Symbol(spec.name) in element for element in matrix) and
                not getattr(spec, 'constant', False)):
                raise ValueError(('The coefficient matrix for the equations '
                                 'contains "%s", which is not constant.') %
                                 spec.name)
        
        symbols = [Symbol(variable) for variable in variables]
        solution = sp.solve_linear_system(matrix.row_join(constants), *symbols)
        b = sp.Matrix([solution[symbol] for symbol in symbols]).transpose()
        
        # Solve the system
        dt = Symbol('dt')    
        A = (matrix * dt).exp()                
        C = sp.Matrix([A.dot(b)]) - b
        S = sp.MatrixSymbol('_S', len(variables), 1)
        updates = A * S + C.transpose()
        
        # The solution contains _S_00, _S_10 etc. for the state variables,
        # replace them with the state variable names 
        abstract_code = []
        for idx, (variable, update) in enumerate(zip(variables, updates)):
            rhs = update.subs('_S_%d0' % idx, variable)
            identifiers = get_identifiers(str(rhs))
            for identifier in identifiers:
                if identifier in specifiers:
                    spec = specifiers[identifier]
                    if isinstance(spec, Value) and spec.scalar and spec.constant:
                        float_val = spec.get_value()
                        rhs = rhs.subs(identifier, float_val)
                elif identifier in namespace:
                    try:
                        float_val = float(namespace[identifier])
                        rhs = rhs.subs(identifier, float_val)
                    except TypeError:
                        # Not a number
                        pass
            abstract_code.append(variable + ' = ' + str(rhs))
        
        return '\n'.join(abstract_code)

linear = LinearStateUpdater()

# The linear state updater has the highest priority
StateUpdateMethod.register('linear', linear, 0)
