'''
Exact integration for linear equations.
'''
import itertools

from sympy import Wild, Symbol
import sympy as sp

from brian2.parsing.sympytools import sympy_to_str
from brian2.utils.logger import get_logger
from brian2.stateupdaters.base import StateUpdateMethod

__all__ = ['linear', 'independent']

logger = get_logger(__name__)

def get_linear_system(eqs):
    '''
    Convert equations into a linear system using sympy.
    
    Parameters
    ----------
    eqs : `Equations`
        The model equations.
    
    Returns
    -------
    (diff_eq_names, coefficients, constants) : (list of str, `sympy.Matrix`, `sympy.Matrix`)
        A tuple containing the variable names (`diff_eq_names`) corresponding
        to the rows of the matrix `coefficients` and the vector `constants`,
        representing the system of equations in the form M * X + B
    
    Raises
    ------
    ValueError
        If the equations cannot be converted into an M * X + B form.
    '''
    diff_eqs = eqs.substituted_expressions
    diff_eq_names = [name for name, _ in diff_eqs]

    symbols = [Symbol(name, real=True) for name in diff_eq_names]

    coefficients = sp.zeros(len(diff_eq_names))
    constants = sp.zeros(len(diff_eq_names), 1)

    for row_idx, (name, expr) in enumerate(diff_eqs):
        s_expr = expr.sympy_expr.expand()

        current_s_expr = s_expr
        for col_idx, symbol in enumerate(symbols):
            current_s_expr = current_s_expr.collect(symbol)
            constant_wildcard = Wild('c', exclude=[symbol])
            factor_wildcard = Wild('c_'+name, exclude=symbols)
            one_pattern = factor_wildcard*symbol + constant_wildcard
            matches = current_s_expr.match(one_pattern)
            if matches is None:
                raise ValueError(('The expression "%s", defining the variable %s, '
                                 'could not be separated into linear components') %
                                 (expr, name))

            coefficients[row_idx, col_idx] = matches[factor_wildcard]
            current_s_expr = matches[constant_wildcard]

        # The remaining constant should be a true constant
        constants[row_idx] = current_s_expr

    return (diff_eq_names, coefficients, constants)


def _non_constant_symbols(symbols, variables, t_symbol):
    '''
    Determine whether the given `sympy.Matrix` only refers to constant
    variables. Note that variables that are not present in the `variables`
    dictionary are considered to be external variables and therefore constant.

    Parameters
    ----------

    symbols : set of `Symbol`
        The symbols to check, e.g. resulting from expression.atoms()
    variables : dict
        The dictionary of `Variable` objects.
    t_symbol : `Symbol`
        The symbol referring to time ``t`` -- will not be considered as
        non-constant in this context because it will specifically checked
        later with `_check_t` (properly taking care of functions that are
        locally constant over a single time step).
    Returns
    -------
    non_constant : set
        A set of non-constant symbols.
    '''
    # As every symbol in the matrix should be either in the namespace or
    # the variables dictionary, it should be sufficient to just check for
    # the presence of any non-constant variables.

    # Only check true symbols, not numbers
    symbols = set([str(symbol) for symbol in symbols
                   if isinstance(symbol, Symbol) and not symbol == t_symbol])

    non_constant = set()

    for symbol in symbols:
        if symbol in variables and not getattr(variables[symbol],
                                                'constant', False):
            non_constant |= {symbol}

    return non_constant


class IndependentStateUpdater(StateUpdateMethod):
    '''
    A state update for equations that do not depend on other state variables,
    i.e. 1-dimensional differential equations. The individual equations are
    solved by sympy.
    '''
    def can_integrate(self, equations, variables):
        if equations.is_stochastic:
            return False

        # Not very efficient but guaranteed to give the correct answer:
        # Just try to apply the integration method
        try:
            self.__call__(equations, variables)
        except (ValueError, NotImplementedError, TypeError) as ex:
            logger.debug('Cannot use independent integration: %s' % ex)
            return False

        # It worked
        return True

    def __call__(self, equations, variables=None):
        if variables is None:
            variables = {}

        if equations.is_stochastic:
            raise ValueError('Cannot solve stochastic equations with this state updater')

        diff_eqs = equations.substituted_expressions

        t = Symbol('t', real=True, positive=True)
        dt = Symbol('dt', real=True, positive=True)
        t0 = Symbol('t0', real=True, positive=True)
        f0 = Symbol('f0', real=True)
        # TODO: Shortcut for simple linear equations? Is all this effort really
        #       worth it?

        code = []
        for name, expression in diff_eqs:
            rhs = expression.sympy_expr
            non_constant = _non_constant_symbols(rhs.atoms(), variables, t) - {
            name}
            if len(non_constant):
                raise ValueError(('Equation for %s referred to non-constant '
                                  'variables %s') % (name, str(non_constant)))
            # We have to be careful and use the real=True assumption as well,
            # otherwise sympy doesn't consider the symbol a match to the content
            # of the equation
            var = Symbol(name, real=True)
            f = sp.Function(name)
            rhs = rhs.subs(var, f(t))
            derivative = sp.Derivative(f(t), t)
            diff_eq = sp.Eq(derivative, rhs)
            # TODO: simplify=True sometimes fails with 0.7.4, see:
            # https://github.com/sympy/sympy/issues/2666
            try:
                general_solution = sp.dsolve(diff_eq, f(t), simplify=True)
            except RuntimeError:
                general_solution = sp.dsolve(diff_eq, f(t), simplify=False)
            # Check whether this is an explicit solution
            if not getattr(general_solution, 'lhs', None) == f(t):
                raise ValueError('Cannot explicitly solve: ' + str(diff_eq))
            # seems to happen sometimes in sympy 0.7.5
            if getattr(general_solution, 'rhs', None) == sp.nan:
                raise ValueError('Cannot explicitly solve: ' + str(diff_eq))
            # Solve for C1 (assuming "var" as the initial value and "t0" as time)
            if general_solution.has(Symbol('C1')):
                if general_solution.has(Symbol('C2')):
                    raise ValueError('Too many constants in solution: %s' % str(general_solution))
                constant_solution = sp.solve(general_solution, Symbol('C1'))
                if len(constant_solution) != 1:
                    raise ValueError(("Couldn't solve for the constant "
                                      "C1 in : %s ") % str(general_solution))
                constant = constant_solution[0].subs(t, t0).subs(f(t0), var)
                solution = general_solution.rhs.subs('C1', constant)
            else:
                solution = general_solution.rhs.subs(t, t0).subs(f(t0), var)
            # Evaluate the expression for one timestep
            solution = solution.subs(t, t + dt).subs(t0, t)
            # only try symplifying it -- it sometimes raises an error
            try:
                solution = solution.simplify()
            except ValueError:
                pass

            code.append(name + ' = ' + sympy_to_str(solution))

        return '\n'.join(code)


def _check_for_locally_constant(expression, variables, dt_value, t_symbol):

    for arg in expression.args:
        if arg is t_symbol:
            # We found "t" -- if it is not the only argument of a locally
            # constant function we bail out
            func_name = str(expression.func)
            if not (func_name in variables and
                        variables[func_name].is_locally_constant(dt_value)):
                raise ValueError(('t is used in a context where we cannot'
                                  'guarantee that it can be considered '
                                  'locally constant.'))
        else:
            _check_for_locally_constant(arg, variables, dt_value, t_symbol)


class LinearStateUpdater(StateUpdateMethod):    
    '''
    A state updater for linear equations. Derives a state updater step from the
    analytical solution given by sympy. Uses the matrix exponential (which is
    only implemented for diagonalizable matrices in sympy).
    ''' 
    def can_integrate(self, equations, variables):
        if equations.is_stochastic:
            return False
               
        # Not very efficient but guaranteed to give the correct answer:
        # Just try to apply the integration method
        try:
            self.__call__(equations, variables)
        except (ValueError, NotImplementedError, TypeError) as ex:
            logger.debug('Cannot use linear integration: %s' % ex)
            return False
        
        # It worked
        return True

    def __call__(self, equations, variables=None):
        
        if variables is None:
            variables = {}
        
        # Get a representation of the ODE system in the form of
        # dX/dt = M*X + B
        varnames, matrix, constants = get_linear_system(equations)

        # Make sure that the matrix M is constant, i.e. it only contains
        # external variables or constant variables
        t = Symbol('t', real=True, positive=True)
        symbols = set.union(*(el.atoms() for el in matrix))
        non_constant = _non_constant_symbols(symbols, variables, t)
        if len(non_constant):
            raise ValueError(('The coefficient matrix for the equations '
                              'contains the symbols %s, which are not '
                              'constant.') % str(non_constant))

        # Check for time dependence
        dt_var = variables.get('dt', None)
        if dt_var is not None:
            # This will raise an error if we meet the symbol "t" anywhere
            # except as an argument of a locally constant function
            for entry in itertools.chain(matrix, constants):
                _check_for_locally_constant(entry, variables, dt_var.get_value(), t)
        symbols = [Symbol(variable, real=True) for variable in varnames]
        solution = sp.solve_linear_system(matrix.row_join(constants), *symbols)
        b = sp.ImmutableMatrix([solution[symbol] for symbol in symbols]).transpose()
        
        # Solve the system
        dt = Symbol('dt', real=True, positive=True)
        A = (matrix * dt).exp()                
        C = sp.ImmutableMatrix([A.dot(b)]) - b
        _S = sp.MatrixSymbol('_S', len(varnames), 1)
        updates = A * _S + C.transpose()
        try:
            # In sympy 0.7.3, we have to explicitly convert it to a single matrix
            # In sympy 0.7.2, it is already a matrix (which doesn't have an
            # is_explicit method)
            updates = updates.as_explicit()
        except AttributeError:
            pass
        
        # The solution contains _S[0, 0], _S[1, 0] etc. for the state variables,
        # replace them with the state variable names 
        abstract_code = []
        for idx, (variable, update) in enumerate(zip(varnames, updates)):
            rhs = update
            for row_idx, varname in enumerate(varnames):
                rhs = rhs.subs(_S[row_idx, 0], varname)

            # Do not overwrite the real state variables yet, the update step
            # of other state variables might still need the original values
            abstract_code.append('_' + variable + ' = ' + sympy_to_str(rhs))
        
        # Update the state variables
        for variable in varnames:
            abstract_code.append('{variable} = _{variable}'.format(variable=variable))
        return '\n'.join(abstract_code)

    def __repr__(self):
        return '%s()' % self.__class__.__name__

independent = IndependentStateUpdater()
linear = LinearStateUpdater()
