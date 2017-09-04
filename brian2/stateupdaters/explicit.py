'''
Numerical integration functions.
'''

import string
import operator

import sympy
from sympy.core.sympify import SympifyError
from pyparsing import (Literal, Group, Word, ZeroOrMore, Suppress, restOfLine,
                       ParseException)

from brian2.parsing.sympytools import str_to_sympy, sympy_to_str
from .base import (StateUpdateMethod, UnsupportedEquationsException,
                   extract_method_options)

__all__ = ['milstein', 'heun', 'euler', 'rk2', 'rk4', 'ExplicitStateUpdater']


#===============================================================================
# Class for simple definition of explicit state updaters
#===============================================================================

def _symbol(name, positive=None):
    ''' Shorthand for ``sympy.Symbol(name, real=True)``. '''
    return sympy.Symbol(name, real=True, positive=positive)

#: reserved standard symbols
SYMBOLS = {'__x' : _symbol('__x'),
           '__t' : _symbol('__t', positive=True),
           'dt': _symbol('dt', positive=True),
           't': _symbol('t', positive=True),
           '__f' : sympy.Function('__f'),
           '__g' : sympy.Function('__g'),
           '__dW': _symbol('__dW')}


def split_expression(expr):
    '''
    Split an expression into a part containing the function ``f`` and another
    one containing the function ``g``. Returns a tuple of the two expressions
    (as sympy expressions).
    
    Parameters
    ----------
    expr : str
        An expression containing references to functions ``f`` and ``g``.    
    
    Returns
    -------
    (non_stochastic, stochastic) : tuple of sympy expressions
        A pair of expressions representing the non-stochastic (containing
        function-independent terms and terms involving ``f``) and the
        stochastic part of the expression (terms involving ``g`` and/or ``dW``).
    
    Examples
    --------
    >>> split_expression('dt * __f(__x, __t)')
    (dt*__f(__x, __t), None)
    >>> split_expression('dt * __f(__x, __t) + __dW * __g(__x, __t)')
    (dt*__f(__x, __t), __dW*__g(__x, __t))
    >>> split_expression('1/(2*dt**.5)*(__g_support - __g(__x, __t))*(__dW**2)')
    (0, __dW**2*__g_support*dt**(-0.5)/2 - __dW**2*dt**(-0.5)*__g(__x, __t)/2)
    '''
    
    f = SYMBOLS['__f']
    g = SYMBOLS['__g']
    dW = SYMBOLS['__dW']
    # Arguments of the f and g functions
    x_f = sympy.Wild('x_f', exclude=[f, g], real=True)
    t_f = sympy.Wild('t_f', exclude=[f, g], real=True)
    x_g = sympy.Wild('x_g', exclude=[f, g], real=True)
    t_g = sympy.Wild('t_g', exclude=[f, g], real=True)
    
    # Reorder the expression so that f(x,t) and g(x,t) are factored out
    sympy_expr = sympy.sympify(expr, locals=SYMBOLS).expand()
    sympy_expr = sympy.collect(sympy_expr, f(x_f, t_f))
    sympy_expr = sympy.collect(sympy_expr, g(x_g, t_g))
    
    # Constant part, contains neither f, g nor dW
    independent = sympy.Wild('independent', exclude=[f,g,dW], real=True)
    # The exponent of the random number
    dW_exponent = sympy.Wild('dW_exponent', exclude=[f,g,dW,0], real=True)
    # The factor for the random number, not containing the g function
    independent_dW = sympy.Wild('independent_dW', exclude=[f,g,dW], real=True)
    # The factor for the f function
    f_factor = sympy.Wild('f_factor', exclude=[f, g], real=True)
    # The factor for the g function
    g_factor = sympy.Wild('g_factor', exclude=[f, g], real=True)

    match_expr = (independent + f_factor * f(x_f, t_f) +
                  independent_dW  * dW ** dW_exponent + g_factor * g(x_g, t_g))    
    matches = sympy_expr.match(match_expr)
    
    if matches is None:
        raise ValueError(('Expression "%s" in the state updater description '
                          'could not be parsed.' % sympy_expr))
    
    # Non-stochastic part    
    if x_f in matches:
        # Includes the f function
        non_stochastic = matches[independent] + (matches[f_factor]*
                                                 f(matches[x_f], matches[t_f]))
    else:
        # Does not include f, might be 0
        non_stochastic = matches[independent]
    
    # Stochastic part
    if independent_dW in matches and matches[independent_dW] != 0:
        # includes a random variable term with a non-zero factor
        stochastic = (matches[g_factor]*g(matches[x_g], matches[t_g]) +
                      matches[independent_dW] * dW ** matches[dW_exponent])
    elif x_g in matches:
        # Does not include a random variable but the g function
        stochastic = matches[g_factor]*g(matches[x_g], matches[t_g])
    else:
        # Contains neither random variable nor g function --> empty
        stochastic = None

    return (non_stochastic, stochastic)


class ExplicitStateUpdater(StateUpdateMethod):
    '''
    An object that can be used for defining state updaters via a simple
    description (see below). Resulting instances can be passed to the
    ``method`` argument of the `NeuronGroup` constructor. As other state
    updater functions the `ExplicitStateUpdater` objects are callable,
    returning abstract code when called with an `Equations` object.
    
    A description of an explicit state updater consists of a (multi-line)
    string, containing assignments to variables and a final "x_new = ...",
    stating the integration result for a single timestep. The assignments
    can be used to define an arbitrary number of intermediate results and
    can refer to ``f(x, t)`` (the function being integrated, as a function of
    ``x``, the previous value of the state variable and ``t``, the time) and
    ``dt``, the size of the timestep.
    
    For example, to define a Runge-Kutta 4 integrator (already provided as
    `rk4`), use::
    
            k1 = dt*f(x,t)
            k2 = dt*f(x+k1/2,t+dt/2)
            k3 = dt*f(x+k2/2,t+dt/2)
            k4 = dt*f(x+k3,t+dt)
            x_new = x+(k1+2*k2+2*k3+k4)/6
    
    Note that for stochastic equations, the function `f` only corresponds to
    the non-stochastic part of the equation. The additional function `g`
    corresponds to the stochastic part that has to be multiplied with the 
    stochastic variable xi (a standard normal random variable -- if the
    algorithm needs a random variable with a different variance/mean you have
    to multiply/add it accordingly). Equations with more than one
    stochastic variable do not have to be treated differently, the part
    referring to ``g`` is repeated for all stochastic variables automatically.
     
    Stochastic integrators can also make reference to ``dW`` (a normal
    distributed random number with variance ``dt``) and ``g(x, t)``, the
    stochastic part of an equation. A stochastic state updater could therefore
    use a description like::
        
        x_new = x + dt*f(x,t) + g(x, t) * dW
    
    For simplicity, the same syntax is used for state updaters that only support
    additive noise, even though ``g(x, t)`` does not depend on ``x`` or ``t``
    in that case.    
    
    There a some restrictions on the complexity of the expressions (but most
    can be worked around by using intermediate results as in the above Runge-
    Kutta example): Every statement can only contain the functions ``f`` and
    ``g`` once; The expressions have to be linear in the functions, e.g. you
    can use ``dt*f(x, t)`` but not ``f(x, t)**2``.
     
    Parameters
    ----------
    description : str
        A state updater description (see above).
    stochastic : {None, 'additive', 'multiplicative'}
        What kind of stochastic equations this state updater supports: ``None``
        means no support of stochastic equations, ``'additive'`` means only
        equations with additive noise and ``'multiplicative'`` means
        supporting arbitrary stochastic equations.
    
    Raises
    ------
    ValueError
        If the parsing of the description failed.
    
    Notes
    -----
    Since clocks are updated *after* the state update, the time ``t`` used
    in the state update step is still at its previous value. Enumerating the
    states and discrete times, ``x_new = x + dt*f(x, t)`` is therefore
    understood as :math:`x_{i+1} = x_i + dt f(x_i, t_i)`, yielding the correct
    forward Euler integration. If the integrator has to refer to the time at
    the end of the timestep, simply use ``t + dt`` instead of ``t``. 
    
    See also
    --------
    euler, rk2, rk4, milstein
    ''' 
    
    #===========================================================================
    # Parsing definitions
    #===========================================================================
    #: Legal names for temporary variables
    TEMP_VAR = ~Literal('x_new') + Word(string.ascii_letters + '_',
                                        string.ascii_letters +
                                        string.digits + '_').setResultsName('identifier')
    
    #: A single expression
    EXPRESSION = restOfLine.setResultsName('expression')
    
    #: An assignment statement
    STATEMENT = Group(TEMP_VAR + Suppress('=') +
                      EXPRESSION).setResultsName('statement')
    
    #: The last line of a state updater description
    OUTPUT = Group(Suppress(Literal('x_new')) + Suppress('=') + EXPRESSION).setResultsName('output')
    
    #: A complete state updater description
    DESCRIPTION = ZeroOrMore(STATEMENT) + OUTPUT
     
    def __init__(self, description, stochastic=None, custom_check=None):
        self._description = description
        self.stochastic = stochastic
        self.custom_check = custom_check

        try:
            parsed = ExplicitStateUpdater.DESCRIPTION.parseString(description,
                                                                  parseAll=True)
        except ParseException as p_exc:
            ex = SyntaxError('Parsing failed: ' + str(p_exc.msg))
            ex.text = str(p_exc.line)
            ex.offset = p_exc.column
            ex.lineno = p_exc.lineno
            raise ex
 
        self.statements = []
        self.symbols = SYMBOLS.copy()
        for element in parsed:
            expression = str_to_sympy(element.expression)
            # Replace all symbols used in state updater expressions by unique
            # names that cannot clash with user-defined variables or functions
            expression = expression.subs(sympy.Function('f'),
                                         self.symbols['__f'])
            expression = expression.subs(sympy.Function('g'),
                                         self.symbols['__g'])
            symbols = list(expression.atoms(sympy.Symbol))
            unique_symbols = []
            for symbol in symbols:
                if symbol.name == 'dt':
                    unique_symbols.append(symbol)
                else:
                    unique_symbols.append(_symbol('__' + symbol.name))
            for symbol, unique_symbol in zip(symbols, unique_symbols):
                expression = expression.subs(symbol, unique_symbol)

            self.symbols.update(dict(((symbol.name, symbol)
                                      for symbol in unique_symbols)))
            if element.getName() == 'statement':
                self.statements.append(('__'+element.identifier, expression))
            elif element.getName() == 'output':
                self.output = expression
            else:
                raise AssertionError('Unknown element name: %s' %
                                     element.getName())

    def __repr__(self):
        # recreate a description string
        description = '\n'.join(['%s = %s' % (var, expr)
                                 for var, expr in self.statements])
        if len(description):
            description += '\n'
        description += 'x_new = ' + str(self.output)
        r = "{classname}('''{description}''', stochastic={stochastic})"
        return r.format(classname=self.__class__.__name__,
                        description=description,
                        stochastic=repr(self.stochastic))

    def __str__(self):
        s = '%s\n' % self.__class__.__name__
        
        if len(self.statements) > 0:
            s += 'Intermediate statements:\n'
            s += '\n'.join([(var + ' = ' + sympy_to_str(expr))
                            for var, expr in self.statements])
            s += '\n'
            
        s += 'Output:\n'
        s += sympy_to_str(self.output)
        return s

    def _latex(self, *args):
        from sympy import latex, Symbol
        s = [r'\begin{equation}']
        for var, expr in self.statements:      
            expr = expr.subs(Symbol('x'), Symbol('x_t'))      
            s.append(latex(Symbol(var)) + ' = ' + latex(expr) + r'\\')
        expr = self.output.subs(Symbol('x'), 'x_t')
        s.append(r'x_{t+1} = ' + latex(expr))
        s.append(r'\end{equation}')
        return '\n'.join(s)

    def _repr_latex_(self):
        return self._latex()

    def replace_func(self, x, t, expr, temp_vars, eq_symbols,
                     stochastic_variable=None):
        '''
        Used to replace a single occurance of ``f(x, t)`` or ``g(x, t)``:
        `expr` is the non-stochastic (in the case of ``f``) or stochastic
        part (``g``) of the expression defining the right-hand-side of the
        differential equation describing `var`. It replaces the variable
        `var` with the value given as `x` and `t` by the value given for
        `t`. Intermediate variables will be replaced with the appropriate
        replacements as well.

        For example, in the `rk2` integrator, the second step involves the
        calculation of ``f(k/2 + x, dt/2 + t)``.  If `var` is ``v`` and
        `expr` is ``-v / tau``, this will result in ``-(_k_v/2 + v)/tau``.

        Note that this deals with only one state variable `var`, given as
        an argument to the surrounding `_generate_RHS` function.
        '''

        try:
            s_expr = str_to_sympy(str(expr))
        except SympifyError as ex:
            raise ValueError('Error parsing the expression "%s": %s' %
                             (expr, str(ex)))

        for var in eq_symbols:
            # Generate specific temporary variables for the state variable,
            # e.g. '_k_v' for the state variable 'v' and the temporary
            # variable 'k'.
            if stochastic_variable is None:
                temp_var_replacements = dict(((self.symbols[temp_var],
                                               _symbol(temp_var+'_'+var))
                                              for temp_var in temp_vars))
            else:
                temp_var_replacements = dict(((self.symbols[temp_var],
                                               _symbol(temp_var+'_'+var+'_'+stochastic_variable))
                                              for temp_var in temp_vars))
            # In the expression given as 'x', replace 'x' by the variable
            # 'var' and all the temporary variables by their
            # variable-specific counterparts.
            x_replacement = x.subs(self.symbols['__x'], eq_symbols[var])
            x_replacement = x_replacement.subs(temp_var_replacements)

            # Replace the variable `var` in the expression by the new `x`
            # expression
            s_expr = s_expr.subs(eq_symbols[var], x_replacement)

        # If the expression given for t in the state updater description
        # is not just "t" (or rather "__t"), then replace t in the
        # equations by it, and replace "__t" by "t" afterwards.
        if t != self.symbols['__t']:
            s_expr = s_expr.subs(SYMBOLS['t'], t)
            s_expr = s_expr.replace(self.symbols['__t'], SYMBOLS['t'])

        return s_expr

    def _non_stochastic_part(self, eq_symbols, non_stochastic,
                             non_stochastic_expr, stochastic_variable,
                             temp_vars, var):
        non_stochastic_results = []
        if stochastic_variable is None or len(stochastic_variable) == 0:
            # Replace the f(x, t) part
            replace_f = lambda x, t: self.replace_func(x, t, non_stochastic,
                                                  temp_vars, eq_symbols)
            non_stochastic_result = non_stochastic_expr.replace(
                self.symbols['__f'],
                replace_f)
            # Replace x by the respective variable
            non_stochastic_result = non_stochastic_result.subs(
                self.symbols['__x'],
                eq_symbols[var])
            # Replace intermediate variables
            temp_var_replacements = dict((self.symbols[temp_var],
                                          _symbol(temp_var + '_' + var))
                                         for temp_var in temp_vars)
            non_stochastic_result = non_stochastic_result.subs(
                temp_var_replacements)
            non_stochastic_results.append(non_stochastic_result)
        elif isinstance(stochastic_variable, basestring):
            # Replace the f(x, t) part
            replace_f = lambda x, t: self.replace_func(x, t, non_stochastic,
                                                  temp_vars, eq_symbols,
                                                  stochastic_variable)
            non_stochastic_result = non_stochastic_expr.replace(
                self.symbols['__f'],
                replace_f)
            # Replace x by the respective variable
            non_stochastic_result = non_stochastic_result.subs(
                self.symbols['__x'],
                eq_symbols[var])
            # Replace intermediate variables
            temp_var_replacements = dict((self.symbols[temp_var],
                                          _symbol(
                                              temp_var + '_' + var + '_' + stochastic_variable))
                                         for temp_var in temp_vars)

            non_stochastic_result = non_stochastic_result.subs(
                temp_var_replacements)
            non_stochastic_results.append(non_stochastic_result)
        else:
            # Replace the f(x, t) part
            replace_f = lambda x, t: self.replace_func(x, t, non_stochastic,
                                                  temp_vars, eq_symbols)
            non_stochastic_result = non_stochastic_expr.replace(
                self.symbols['__f'],
                replace_f)
            # Replace x by the respective variable
            non_stochastic_result = non_stochastic_result.subs(
                self.symbols['__x'],
                eq_symbols[var])
            # Replace intermediate variables
            temp_var_replacements = dict((self.symbols[temp_var],
                                          reduce(operator.add, [_symbol(
                                              temp_var + '_' + var + '_' + xi)
                                                                for xi in
                                                                stochastic_variable]))
                                         for temp_var in temp_vars)

            non_stochastic_result = non_stochastic_result.subs(
                temp_var_replacements)
            non_stochastic_results.append(non_stochastic_result)

        return non_stochastic_results

    def _stochastic_part(self, eq_symbols, stochastic, stochastic_expr,
                         stochastic_variable, temp_vars, var):
        stochastic_results = []
        if isinstance(stochastic_variable, basestring):
            # Replace the g(x, t) part
            replace_f = lambda x, t: self.replace_func(x, t,
                                                       stochastic.get(stochastic_variable, 0),
                                                       temp_vars, eq_symbols,
                                                       stochastic_variable)
            stochastic_result = stochastic_expr.replace(self.symbols['__g'],
                                                        replace_f)
            # Replace x by the respective variable
            stochastic_result = stochastic_result.subs(self.symbols['__x'],
                                                       eq_symbols[var])
            # Replace dW by the respective variable
            stochastic_result = stochastic_result.subs(self.symbols['__dW'],
                                                       stochastic_variable)
            # Replace intermediate variables
            temp_var_replacements = dict((self.symbols[temp_var],
                                          _symbol(
                                              temp_var + '_' + var + '_' + stochastic_variable))
                                         for temp_var in temp_vars)

            stochastic_result = stochastic_result.subs(temp_var_replacements)
            stochastic_results.append(stochastic_result)
        else:
            for xi in stochastic_variable:
                # Replace the g(x, t) part
                replace_f = lambda x, t: self.replace_func(x, t,
                                                           stochastic.get(xi, 0),
                                                           temp_vars,
                                                           eq_symbols, xi)
                stochastic_result = stochastic_expr.replace(self.symbols['__g'],
                                                            replace_f)
                # Replace x by the respective variable
                stochastic_result = stochastic_result.subs(self.symbols['__x'],
                                                           eq_symbols[var])

                # Replace dW by the respective variable
                stochastic_result = stochastic_result.subs(self.symbols['__dW'],
                                                           xi)


                # Replace intermediate variables
                temp_var_replacements = dict((self.symbols[temp_var],
                                              _symbol(temp_var + '_' + var + '_' + xi))
                                             for temp_var in temp_vars)

                stochastic_result = stochastic_result.subs(
                    temp_var_replacements)
                stochastic_results.append(stochastic_result)
        return stochastic_results

    def _generate_RHS(self, eqs, var, eq_symbols, temp_vars, expr,
                      non_stochastic_expr, stochastic_expr,
                      stochastic_variable=()):
        '''
        Helper function used in `__call__`. Generates the right hand side of
        an abstract code statement by appropriately replacing f, g and t.
        For example, given a differential equation ``dv/dt = -(v + I) / tau``
        (i.e. `var` is ``v` and `expr` is ``(-v + I) / tau``) together with
        the `rk2` step ``return x + dt*f(x +  k/2, t + dt/2)``
        (i.e. `non_stochastic_expr` is
        ``x + dt*f(x +  k/2, t + dt/2)`` and `stochastic_expr` is ``None``),
        produces ``v + dt*(-v - _k_v/2 + I + _k_I/2)/tau``.
                
        '''
        
        # Note: in the following we are silently ignoring the case that a
        # state updater does not care about either the non-stochastic or the
        # stochastic part of an equation. We do trust state updaters to
        # correctly specify their own abilities (i.e. they do not claim to
        # support stochastic equations but actually just ignore the stochastic
        # part). We can't really check the issue here, as we are only dealing
        # with one line of the state updater description. It is perfectly valid
        # to write the euler update as:
        #     non_stochastic = dt * f(x, t)
        #     stochastic = dt**.5 * g(x, t) * xi
        #     return x + non_stochastic + stochastic
        #
        # In the above case, we'll deal with lines which do not define either
        # the stochastic or the non-stochastic part.
        
        non_stochastic, stochastic = expr.split_stochastic()

        if non_stochastic_expr is not None:
            # We do have a non-stochastic part in the state updater description
            non_stochastic_results = self._non_stochastic_part(eq_symbols,
                                                               non_stochastic,
                                                               non_stochastic_expr,
                                                               stochastic_variable,
                                                               temp_vars, var)
        else:
            non_stochastic_results = []

        if not (stochastic is None or stochastic_expr is None):
            # We do have a stochastic part in the state
            # updater description
            stochastic_results = self._stochastic_part(eq_symbols,
                                                       stochastic,
                                                       stochastic_expr,
                                                       stochastic_variable,
                                                       temp_vars, var)
        else:
            stochastic_results = []

        RHS = sympy.Number(0)
        # All the parts (one non-stochastic and potentially more than one
        # stochastic part) are combined with addition
        for non_stochastic_result in non_stochastic_results:
            RHS += non_stochastic_result
        for stochastic_result in stochastic_results:
            RHS += stochastic_result

        return sympy_to_str(RHS)

    def __call__(self, eqs, variables=None, method_options=None):
        '''
        Apply a state updater description to model equations.
        
        Parameters
        ----------
        eqs : `Equations`
            The equations describing the model
        variables: dict-like, optional
            The `Variable` objects for the model. Ignored by the explicit
            state updater.
        method_options : dict, optional
            Additional options to the state updater (not used at the moment
            for the explicit state updaters).

        Examples
        --------
        >>> from brian2 import *
        >>> eqs = Equations('dv/dt = -v / tau : volt')
        >>> print(euler(eqs))
        _v = -dt*v/tau + v
        v = _v
        >>> print(rk4(eqs))
        __k_1_v = -dt*v/tau
        __k_2_v = -dt*(0.5*__k_1_v + v)/tau
        __k_3_v = -dt*(0.5*__k_2_v + v)/tau
        __k_4_v = -dt*(__k_3_v + v)/tau
        _v = 0.166666666666667*__k_1_v + 0.333333333333333*__k_2_v + 0.333333333333333*__k_3_v + 0.166666666666667*__k_4_v + v
        v = _v
        '''
        method_options = extract_method_options(method_options, {})
        # Non-stochastic numerical integrators should work for all equations,
        # except for stochastic equations
        if eqs.is_stochastic:
            if self.stochastic is None:
                raise UnsupportedEquationsException('Cannot integrate '
                                                    'stochastic equations with '
                                                    'this state updater.')
            if (self.stochastic != 'multiplicative' and
                        eqs.stochastic_type == 'multiplicative'):
                raise UnsupportedEquationsException('Cannot integrate '
                                                    'equations with '
                                                    'multiplicative noise with '
                                                    'this state updater.')

        if self.custom_check:
            self.custom_check(eqs, variables)
        # The final list of statements
        statements = []

        stochastic_variables = eqs.stochastic_variables

        # The variables for the intermediate results in the state updater
        # description, e.g. the variable k in rk2
        intermediate_vars = [var for var, expr in self.statements]
        
        # A dictionary mapping all the variables in the equations to their
        # sympy representations 
        eq_variables = dict(((var, _symbol(var)) for var in eqs.eq_names))
        
        # Generate the random numbers for the stochastic variables
        for stochastic_variable in stochastic_variables:
            statements.append(stochastic_variable + ' = ' + 'dt**.5 * randn()')

        substituted_expressions = eqs.get_substituted_expressions(variables)

        # Process the intermediate statements in the stateupdater description
        for intermediate_var, intermediate_expr in self.statements:
                      
            # Split the expression into a non-stochastic and a stochastic part
            non_stochastic_expr, stochastic_expr = split_expression(intermediate_expr)
            
            # Execute the statement by appropriately replacing the functions f
            # and g and the variable x for every equation in the model.
            # We use the model equations where the subexpressions have
            # already been substituted into the model equations.
            for var, expr in substituted_expressions:
                for xi in stochastic_variables:
                    RHS = self._generate_RHS(eqs, var, eq_variables, intermediate_vars,
                                             expr, non_stochastic_expr,
                                             stochastic_expr, xi)
                    statements.append(intermediate_var+'_'+var+'_'+xi+' = '+RHS)
                if not stochastic_variables:   # no stochastic variables
                    RHS = self._generate_RHS(eqs, var, eq_variables, intermediate_vars,
                                             expr, non_stochastic_expr,
                                             stochastic_expr)
                    statements.append(intermediate_var+'_'+var+' = '+RHS)
                
        # Process the "return" line of the stateupdater description
        non_stochastic_expr, stochastic_expr = split_expression(self.output)
        
        # Assign a value to all the model variables described by differential
        # equations
        for var, expr in substituted_expressions:
            RHS = self._generate_RHS(eqs, var, eq_variables, intermediate_vars,
                                     expr, non_stochastic_expr, stochastic_expr,
                                     stochastic_variables)
            statements.append('_' + var + ' = ' + RHS)
        
        # Assign everything to the final variables
        for var, expr in substituted_expressions:
            statements.append(var + ' = ' + '_' + var)

        return '\n'.join(statements)

#===============================================================================
# Excplicit state updaters
#===============================================================================

# these objects can be used like functions because they are callable

#: Forward Euler state updater
euler = ExplicitStateUpdater('x_new = x + dt * f(x,t) + g(x,t) * dW',
                             stochastic='additive')

#: Second order Runge-Kutta method (midpoint method)
rk2 = ExplicitStateUpdater('''
    k = dt * f(x,t)
    x_new = x + dt*f(x +  k/2, t + dt/2)''')

#: Classical Runge-Kutta method (RK4)
rk4 = ExplicitStateUpdater('''
    k_1 = dt*f(x,t)
    k_2 = dt*f(x+k_1/2,t+dt/2)
    k_3 = dt*f(x+k_2/2,t+dt/2)
    k_4 = dt*f(x+k_3,t+dt)
    x_new = x+(k_1+2*k_2+2*k_3+k_4)/6
    ''')

def diagonal_noise(equations, variables):
    '''
    Checks whether we deal with diagonal noise, i.e. one independent noise
    variable per variable.

    Raises
    ------
    UnsupportedEquationsException
        If the noise is not diagonal.
    '''
    if not equations.is_stochastic:
        return

    stochastic_vars = []
    for _, expr in equations.get_substituted_expressions(variables):
        expr_stochastic_vars = expr.stochastic_variables
        if len(expr_stochastic_vars) > 1:
            # More than one stochastic variable --> no diagonal noise
            raise UnsupportedEquationsException('Cannot integrate stochastic '
                                                'equations with non-diagonal '
                                                'noise with this state '
                                                'updater.')
        stochastic_vars.extend(expr_stochastic_vars)

    # If there's no stochastic variable is used in more than one equation, we
    # have diagonal noise
    if len(stochastic_vars) != len(set(stochastic_vars)):
        raise UnsupportedEquationsException('Cannot integrate stochastic '
                                            'equations with non-diagonal '
                                            'noise with this state '
                                            'updater.')

#: Derivative-free Milstein method
milstein = ExplicitStateUpdater('''
    x_support = x + dt*f(x, t) + dt**.5 * g(x, t)
    g_support = g(x_support, t)
    k = 1/(2*dt**.5)*(g_support - g(x, t))*(dW**2)
    x_new = x + dt*f(x,t) + g(x, t) * dW + k
    ''', stochastic='multiplicative', custom_check=diagonal_noise)

#: Stochastic Heun method (for multiplicative Stratonovic SDEs with non-diagonal
#: diffusion matrix)
heun = ExplicitStateUpdater('''
    x_support = x + g(x,t) * dW
    g_support = g(x_support,t+dt)
    x_new = x + dt*f(x,t) + .5*dW*(g(x,t)+g_support)
    ''', stochastic='multiplicative')
