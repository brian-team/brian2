'''
Exact integration for linear equations.
'''

import itertools

import sympy as sp
from sympy import Wild, Symbol, I, re, im

from brian2.equations.codestrings import is_constant_over_dt
from brian2.parsing.sympytools import sympy_to_str, str_to_sympy
from brian2.stateupdaters.base import (StateUpdateMethod,
                                       UnsupportedEquationsException,
                                       extract_method_options)
from brian2.utils.logger import get_logger

__all__ = ['linear', 'exact', 'independent']

logger = get_logger(__name__)


def get_linear_system(eqs, variables):
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
    diff_eqs = {name: str_to_sympy(expr.code, variables).expand()
                for name, expr in eqs.get_substituted_expressions(variables)}

    # Sometimes, in particular in testing, variables defined as differential
    # equations are actually constant (e.g. `dv/dt = 0/second`). We ignore
    # them here
    symbols = [Symbol(name, real=True) for name, expr in diff_eqs.items()
               if expr != 0]

    coefficients = sp.zeros(len(symbols))
    constants = sp.zeros(len(symbols), 1)

    for row_idx, symbol in enumerate(symbols):
        s_expr = diff_eqs[symbol.name]
        current_s_expr = s_expr
        for col_idx, symbol in enumerate(symbols):
            current_s_expr = current_s_expr.collect(symbol)
            constant_wildcard = Wild('c', exclude=[symbol])
            factor_wildcard = Wild('c_'+symbol.name, exclude=symbols)
            one_pattern = factor_wildcard*symbol + constant_wildcard
            matches = current_s_expr.match(one_pattern)
            if matches is None:
                raise UnsupportedEquationsException(('The expression "%s", '
                                                     'defining the variable '
                                                     '%s, could not be '
                                                     'separated into linear '
                                                     'components.') %
                                                    (sympy_to_str(s_expr),
                                                     symbol.name))

            coefficients[row_idx, col_idx] = matches[factor_wildcard]
            current_s_expr = matches[constant_wildcard]

        # The remaining constant should be a true constant
        constants[row_idx] = current_s_expr

    return [s.name for s in symbols], coefficients, constants


class IndependentStateUpdater(StateUpdateMethod):
    '''
    A state update for equations that do not depend on other state variables,
    i.e. 1-dimensional differential equations. The individual equations are
    solved by sympy.

    .. deprecated:: 2.1
        This method might be removed from future versions of Brian.
    '''

    def __call__(self, equations, variables=None, method_options=None):
        logger.warn("The 'independent' state updater is deprecated and might be "
                    "removed in future versions of Brian.",
                    'deprecated_independent', once=True)
        extract_method_options(method_options, {})
        if equations.is_stochastic:
            raise UnsupportedEquationsException('Cannot solve stochastic '
                                                'equations with this state '
                                                'updater')
        if variables is None:
            variables = {}

        diff_eqs = equations.get_substituted_expressions(variables)

        t = Symbol('t', real=True, positive=True)
        dt = Symbol('dt', real=True, positive=True)
        t0 = Symbol('t0', real=True, positive=True)

        code = []
        for name, expression in diff_eqs:
            rhs = str_to_sympy(expression.code, variables)

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
                raise UnsupportedEquationsException('Cannot explicitly solve: '
                                                    + str(diff_eq))
            # Solve for C1 (assuming "var" as the initial value and "t0" as time)
            if general_solution.has(Symbol('C1')):
                if general_solution.has(Symbol('C2')):
                    raise UnsupportedEquationsException('Too many constants in solution: %s' % str(general_solution))
                constant_solution = sp.solve(general_solution, Symbol('C1'))
                if len(constant_solution) != 1:
                    raise UnsupportedEquationsException(("Couldn't solve for the constant "
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


class LinearStateUpdater(StateUpdateMethod):
    '''
    A state updater for linear equations. Derives a state updater step from the
    analytical solution given by sympy. Uses the matrix exponential (which is
    only implemented for diagonalizable matrices in sympy).
    '''
    def __call__(self, equations, variables=None, method_options=None):
        method_options = extract_method_options(method_options,
                                                {'simplify': True})

        if equations.is_stochastic:
            raise UnsupportedEquationsException('Cannot solve stochastic '
                                                'equations with this state '
                                                'updater.')
        if variables is None:
            variables = {}

        # Get a representation of the ODE system in the form of
        # dX/dt = M*X + B
        varnames, matrix, constants = get_linear_system(equations, variables)

        # No differential equations, nothing to do (this occurs sometimes in the
        # test suite where the whole model is nothing more than something like
        # 'v : 1')
        if matrix.shape == (0, 0):
            return ''

        # Make sure that the matrix M is constant, i.e. it only contains
        # external variables or constant variables

        # Check for time dependence
        dt_value = variables['dt'].get_value()[0] if 'dt' in variables else None

        # This will raise an error if we meet the symbol "t" anywhere
        # except as an argument of a locally constant function
        for entry in itertools.chain(matrix, constants):
            if not is_constant_over_dt(entry, variables, dt_value):
                raise UnsupportedEquationsException(
                    ('Expression "{}" is not guaranteed to be constant over a '
                     'time step').format(sympy_to_str(entry)))

        # Solve the system
        dt = Symbol('dt', real=True, positive=True)
        # Add the constant terms as new variables
        const_vars = []
        const_terms = []
        for idx, (varname, const_term) in enumerate(zip(varnames, constants)):
            if const_term != 0:
                matrix = matrix.col_insert(matrix.cols, sp.Matrix([1 if i == idx else 0
                                                                   for i in range(matrix.rows)]))
                matrix = matrix.row_insert(matrix.rows, sp.zeros(1, matrix.cols))
                const_vars.append('_const_term_' + varname)
                const_terms.append(const_term)

        try:
            A = (matrix * dt).exp()
        except NotImplementedError:
            raise UnsupportedEquationsException('Cannot solve the given '
                                                'equations with this '
                                                'stateupdater.')

        if method_options['simplify']:
            A = A.applyfunc(lambda x:
                            sp.factor_terms(sp.cancel(sp.signsimp(x))))

        _S = sp.MatrixSymbol('_S', len(varnames) + len(const_vars), 1)
        updates = A * _S
        updates = updates.as_explicit()
        abstract_code = []

        # Add code for the constant terms:
        for const_var, const_term in zip(const_vars, const_terms):
            abstract_code.append(const_var + ' = ' + sympy_to_str(const_term))

        # The solution contains _S[0, 0], _S[1, 0] etc. for the state variables,
        # replace them with the state variable names
        for variable, update in zip(varnames, updates[:len(varnames)]):
            rhs = update
            if rhs.has(I, re, im):
                raise UnsupportedEquationsException('The solution to the linear system '
                                                    'contains complex values '
                                                    'which is currently not implemented.')

            for row_idx, varname in enumerate(itertools.chain(varnames, const_vars)):
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
exact = linear
