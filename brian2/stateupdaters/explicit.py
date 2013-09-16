'''
Numerical integration functions.
'''

import string

import sympy
from sympy.core.sympify import SympifyError
from pyparsing import (Literal, Group, Word, ZeroOrMore, Suppress, restOfLine,
                       ParseException)

from brian2.parsing.sympytools import str_to_sympy, sympy_to_str

from .base import StateUpdateMethod

__all__ = ['milstein', 'euler', 'rk2', 'rk4', 'ExplicitStateUpdater']


#===============================================================================
# Class for simple definition of explicit state updaters
#===============================================================================

def _symbol(name):
    ''' Shorthand for ``sympy.Symbol(name, real=True)``. '''
    return sympy.Symbol(name, real=True)

#: reserved standard symbols
SYMBOLS = {'x' : _symbol('x'),
           't' : _symbol('t'),
           'dt': _symbol('dt'),
           'f' : sympy.Function('f'),
           'g' : sympy.Function('g'),
           'dW': _symbol('dW')}


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
    >>> split_expression('dt * f(x, t)')
    (dt*f(x, t), None)
    >>> split_expression('dt * f(x, t) + dW * g(x, t)')
    (dt*f(x, t), dW*g(x, t))
    >>> split_expression('1/(2*dt**.5)*(g_support - g(x, t))*(dW**2)')
    (0, dW**2*dt**(-0.5)*g_support/2 - dW**2*dt**(-0.5)*g(x, t)/2)
    '''
    
    f = SYMBOLS['f']
    g = SYMBOLS['g']
    dW = SYMBOLS['dW']
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
     
    def __init__(self, description, stochastic=None):
        self._description = description
        self.stochastic = stochastic
                
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
            symbols = list(expression.atoms(sympy.Symbol))
            self.symbols.update(dict(((symbol.name, symbol)
                                      for symbol in symbols)))
            if element.getName() == 'statement':
                self.statements.append((element.identifier, expression))
            elif element.getName() == 'output':
                self.output = expression
            else:
                raise AssertionError('Unknown element name: %s' %
                                     element.getName())
    
    def can_integrate(self, equations, variables):
        # Non-stochastic numerical integrators should work for all equations,
        # except for stochastic equations
        if equations.is_stochastic and self.stochastic is None:
            return False
        elif (equations.stochastic_type == 'multiplicative' and
              self.stochastic != 'multiplicative'):
            return False
        else:
            return True
    
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

    def _generate_RHS(self, eqs, var, eq_symbols, temp_vars, expr,
                      non_stochastic_expr, stochastic_expr):
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
        
        def replace_func(x, t, expr, temp_vars):
            '''
            Used to replace a single occurance of ``f(x, t)`` or ``g(x, t)``:
            `expr` is the non-stochastic (in the case of ``f``) or stochastic
            part (``g``) of the expression defining the right-hand-side of the
            differential equation describing `var`. It replaces the variable
            `var` with the value given as `x` and `t` by the value given for
            `t. Intermediate variables will be replaced with the appropriate
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
                temp_var_replacements = dict(((self.symbols[temp_var],
                                               _symbol('_'+temp_var+'_'+var))
                                              for temp_var in temp_vars))
                # In the expression given as 'x', replace 'x' by the variable
                # 'var' and all the temporary variables by their
                # variable-specific counterparts.
                x_replacement = x.subs(self.symbols['x'], eq_symbols[var])
                x_replacement = x_replacement.subs(temp_var_replacements)
                
                # Replace the variable `var` in the expression by the new `x`
                # expression
                s_expr = s_expr.subs(eq_symbols[var], x_replacement)
                
            # Directly substitute the 't' expression for the symbol t, there
            # are no temporary variables to consider here.             
            s_expr = s_expr.subs(self.symbols['t'], t)
            
            return s_expr
        
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
        # We do have a non-stochastic part in our equation and in the state
        # updater description 
        if not (non_stochastic is None or non_stochastic_expr is None):
            # Replace the f(x, t) part
            replace_f = lambda x, t:replace_func(x, t, non_stochastic,
                                                 temp_vars)
            non_stochastic_result = non_stochastic_expr.replace(self.symbols['f'],
                                                                replace_f)
            # Replace x by the respective variable
            non_stochastic_result = non_stochastic_result.subs(self.symbols['x'],
                                                               eq_symbols[var])
            # Replace intermediate variables
            temp_var_replacements = dict((self.symbols[temp_var],
                                           _symbol('_'+temp_var+'_'+var))
                                         for temp_var in temp_vars)
            non_stochastic_result = non_stochastic_result.subs(temp_var_replacements)
        else:
            non_stochastic_result = None

        # We do have a stochastic part in our equation and in the state updater
        # description
        if not (stochastic is None or stochastic_expr is None):
            stochastic_results = []
            
            # We potentially have more than one stochastic variable
            for xi in stochastic:
                # Replace the g(x, t)*xi part
                replace_g = lambda x, t:replace_func(x, t, stochastic[xi],
                                                     temp_vars)
                stochastic_result = stochastic_expr.replace(self.symbols['g'],
                                                            replace_g)
                
                # Replace x and xi by the respective variables
                stochastic_result = stochastic_result.subs(self.symbols['x'],
                                                           eq_symbols[var])
                stochastic_result = stochastic_result.subs(self.symbols['dW'], xi)   

                # Replace intermediate variables
                temp_var_replacements = dict((self.symbols[temp_var],
                                               _symbol('_'+temp_var+'_'+var))
                                             for temp_var in temp_vars)
                
                stochastic_result = stochastic_result.subs(temp_var_replacements)

                stochastic_results.append(stochastic_result)                        
        else:
            stochastic_results = []
        
        RHS = []
        # All the parts (one non-stochastic and potentially more than one
        # stochastic part) are combined with addition
        if non_stochastic_result is not None:
            RHS.append(sympy_to_str(non_stochastic_result))
        for stochastic_result in stochastic_results:
            RHS.append(sympy_to_str(stochastic_result))
        
        RHS = ' + '.join(RHS)
        return RHS


    def __call__(self, eqs, variables=None):
        '''
        Apply a state updater description to model equations.
        
        Parameters
        ----------
        eqs : `Equations`
            The equations describing the model

        
        variables: dict-like, optional
            The `Variable` objects for the model. Ignored by the explicit
            state updater.
        
        Examples
        --------
        >>> from brian2 import *
        >>> eqs = Equations('dv/dt = -v / tau : volt')        
        >>> print(euler(eqs))
        _v = -dt*v/tau + v
        v = _v
        >>> print(rk4(eqs))
        _k_1_v = -dt*v/tau
        _k_2_v = -dt*(0.5*_k_1_v + v)/tau
        _k_3_v = -dt*(0.5*_k_2_v + v)/tau
        _k_4_v = -dt*(_k_3_v + v)/tau
        _v = 0.166666666666667*_k_1_v + 0.333333333333333*_k_2_v + 0.333333333333333*_k_3_v + 0.166666666666667*_k_4_v + v
        v = _v
        '''
        
        # The final list of statements
        statements = []
        
        # The variables for the intermediate results in the state updater
        # description, e.g. the variable k in rk2
        intermediate_vars = [var for var, expr in self.statements]
        
        # A dictionary mapping all the variables in the equations to their
        # sympy representations 
        eq_variables = dict(((var, _symbol(var)) for var in eqs.eq_names))
        
        # Generate the random numbers for the stochastic variables
        stochastic_variables = eqs.stochastic_variables
        for stochastic_variable in stochastic_variables:
            statements.append(stochastic_variable + ' = ' + 'dt**.5 * randn()')
        
        # Process the intermediate statements in the stateupdater description
        for intermediate_var, intermediate_expr in self.statements:
                      
            # Split the expression into a non-stochastic and a stochastic part
            non_stochastic_expr, stochastic_expr = split_expression(intermediate_expr)
            
            # Execute the statement by appropriately replacing the functions f
            # and g and the variable x for every equation in the model.
            # We use the model equations where the static equations have
            # already been substituted into the model equations.
            for var, expr in eqs.substituted_expressions:
                RHS = self._generate_RHS(eqs, var, eq_variables, intermediate_vars,
                                         expr, non_stochastic_expr,
                                         stochastic_expr)                
                statements.append('_'+intermediate_var+'_'+var+' = '+RHS)
                
        # Process the "return" line of the stateupdater description
        non_stochastic_expr, stochastic_expr = split_expression(self.output)
        
        # Assign a value to all the model variables described by differential
        # equations       
        for var, expr in eqs.substituted_expressions:
            RHS = self._generate_RHS(eqs, var, eq_variables, intermediate_vars,
                                     expr, non_stochastic_expr, stochastic_expr)
            statements.append('_' + var + ' = ' + RHS)
        
        # Assign everything to the final variables
        for var, expr in eqs.substituted_expressions:
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

#: Derivative-free Milstein method
milstein = ExplicitStateUpdater('''
    x_support = x + dt*f(x, t) + dt**.5 * g(x, t)
    g_support = g(x_support, t)
    k = 1/(2*dt**.5)*(g_support - g(x, t))*(dW**2)
    x_new = x + dt*f(x,t) + g(x, t) * dW + k
    ''', stochastic='multiplicative')
