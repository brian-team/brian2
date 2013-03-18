'''
Numerical integration functions.
'''

import string

import sympy
from pyparsing import (Literal, Group, Word, ZeroOrMore, Suppress, restOfLine,
                       ParseException)

from brian2.utils.parsing import parse_to_sympy

from .base import StateUpdateMethod
from sympy.core.sympify import SympifyError

__all__ = ['milstein', 'euler', 'rk2', 'rk4', 'ExplicitStateUpdater']

#===============================================================================
# Parsing definitions
#===============================================================================
TEMP_VAR = Word(string.ascii_letters + '_',
                string.ascii_letters + string.digits + '_').setResultsName('identifier')

EXPRESSION = restOfLine.setResultsName('expression')

STATEMENT = Group(TEMP_VAR + Suppress('=') + EXPRESSION).setResultsName('statement')

OUTPUT = Group(Suppress(Literal('return ')) + EXPRESSION).setResultsName('output')

DESCRIPTION = ZeroOrMore(STATEMENT) + OUTPUT 

#===============================================================================
# Class for simple definition of explicit state updaters
#===============================================================================

# reserved standard symbols
SYMBOLS = {'x' : sympy.Symbol('x'),
           't' : sympy.Symbol('t'),
           'dt': sympy.Symbol('dt'),
           'f' : sympy.Function('f'),
           'g' : sympy.Function('g'),
           'dW': sympy.Symbol('dW')}

def split_expression(expr):
    '''
    Split an expression into a part containing the function ``f`` and another
    one containing the function ``g``. Returns a tuple of the two expressions
    (as strings).
    
    Parameters
    ----------
    expr : str
        An expression containing references to functions ``f`` and ``g``.    
    
    Returns
    -------
    (non_stochastic, stochastic) : tuple of sympy expressions
        A pair of sympy expressions representing the non-stochastic (containing
        function-independent terms and terms involving ``f``) and the
        stochastic part of the expression (terms involving ``g`` and/or ``xi``).
    
    Examples
    --------
    TODO
    '''
    
    f = SYMBOLS['f']
    g = SYMBOLS['g']
    dW = SYMBOLS['dW']
    x_f = sympy.Wild('x_f', exclude=[f, g])
    t_f = sympy.Wild('t_f', exclude=[f, g])
    x_g = sympy.Wild('x_g', exclude=[f, g])
    t_g = sympy.Wild('t_g', exclude=[f, g])
    
    sympy_expr = sympy.sympify(expr, locals=SYMBOLS).expand()
    sympy_expr = sympy.collect(sympy_expr, f(x_f, t_f))
    sympy_expr = sympy.collect(sympy_expr, g(x_g, t_g))
    
    independent = sympy.Wild('independent', exclude=[f,g,dW])
    dW_exponent = sympy.Wild('dW_exponent', exclude=[f,g,dW,0])
    independent_dW = sympy.Wild('independent_dW', exclude=[f,g,dW])
    f_factor = sympy.Wild('f_factor', exclude=[f, g])
    g_factor = sympy.Wild('g_factor', exclude=[f, g])    

    match_expr = (independent + f_factor * f(x_f, t_f) +
                  independent_dW  * dW ** dW_exponent + g_factor * g(x_g, t_g))    
    matches = sympy_expr.match(match_expr)
    
    if matches is None:
        raise ValueError(('Expression "%s" in the state updater description '
                          'could not be parsed.' % sympy_expr))
    
    if x_f in matches:
        non_stochastic = matches[independent] + matches[f_factor]*f(matches[x_f], matches[t_f])
    else:
        non_stochastic = matches[independent]
        
    if independent_dW in matches and matches[independent_dW] != 0:
        stochastic = (matches[g_factor]*g(matches[x_g], matches[t_g]) +
                      matches[independent_dW] * dW ** matches[dW_exponent])
    elif x_g in matches:
        stochastic = matches[g_factor]*g(matches[x_g], matches[t_g])
    else:
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
    string, containing assignments to variables and a final return line,
    returning the integration result for a single timestep. The assignments
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
            return x+(k1+2*k2+2*k3+k4)/6
    
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
    stochastic part of an equation. A stochastic state updater good therefore
    use a description like:     
        
        return x + dt*f(x,t) + g(x, t) * dW
    
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
    priority : int
        The priority of this state updater (in case it is applicable in
        general). Higher values mean that it is more likely to be chosen.
    stochastic : {None, 'additive', 'multiplicative'}
        What kind of stochastic equations this state updater supports: ``None``
        means no support of stochastic equations, ``'additive'`` means only
        equations with additive noise and ``"multiplicative'`` means
        supporting arbitrary stochastic equations.
    
    Raises
    ------
    ValueError
        If the parsing of the description failed.
    
    See also
    --------
    euler, rk2, rk4
    ''' 
    
    def __init__(self, description, priority, stochastic=None):
        self.priority = priority
        self.stochastic = stochastic
                
        try:
            parsed = DESCRIPTION.parseString(description, parseAll=True)
        except ParseException as p_exc:
            raise ValueError('Parsing failed: \n' + str(p_exc.line) + '\n' +
                              ' '*(p_exc.column - 1) + '^\n' + str(p_exc))
 
        self.statements = []
        self.symbols = SYMBOLS.copy()
        for element in parsed:
            # Make sure to always re-use symbol objects for known symbols,
            # otherwise the replacements go wrong
            expression = parse_to_sympy(element.expression,
                                        local_dict=self.symbols)
            symbols = list(expression.atoms(sympy.Symbol))
            self.symbols.update(dict([(symbol.name, symbol)
                                      for symbol in symbols]))
            if element.getName() == 'statement':
                self.statements.append((element.identifier, expression))
            elif element.getName() == 'output':
                self.output = expression
            else:
                raise AssertionError('Unknown element name: %s' % element.getName())
    
    def get_priority(self, equations, namespace, specifiers):
        # Non-stochastic numerical integrators should work for all equations,
        # except for stochastic equations
        if equations.is_stochastic and self.stochastic is None:
            return 0
        if (equations.stochastic_type == 'multiplicative' and
            self.stochastic != 'multiplicative'):
            return 0
        else:
            return self.priority
    
    def __str__(self):
        s = ''
        
        if len(self.statements) > 0:
            s += 'Intermediate statements:\n'
            s += '\n'.join([(var + ' = ' + str(expr)) for var, expr in self.statements])
            s += '\n'
            
        s += 'Output:\n'
        s += str(self.output)
        return s

    def _generate_RHS(self, eqs, var, symbols, temp_vars, expr,
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
            Replace an occurance of ``f(x, t)`` or ``g(x, t)`` in the given
            expression `expr`, where ``x`` will be replaced with the
            variable `var` and any intermediate variables with the appropriate
            replacements (see `_generate_RHS`).
            '''
            try:
                s_expr = parse_to_sympy(expr, local_dict=symbols)
            except SympifyError as ex:
                raise ValueError('Error parsing the expression "%s": %s' %
                                 (expr, str(ex)))
            
            for var in eqs.diff_eq_names:
                temp_vars_specific = dict([('_' + temp_var + '_' + var,
                                            sympy.Symbol('_' + temp_var + '_' + var))
                                           for temp_var in temp_vars])                
                symbols.update(temp_vars_specific)
                temp_var_replacements = dict([(temp_var, temp_vars_specific['_' + temp_var + '_' + var])
                                              for temp_var in temp_vars])
                one_replacement = x.subs(symbols['x'], symbols[var])
                                
                one_replacement = one_replacement.subs(temp_var_replacements)
                
                s_expr = s_expr.subs(symbols[var], one_replacement)
            
            # replace time (important for time-dependent equations)
            s_expr = s_expr.subs(symbols['t'], t)
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
            non_stochastic_result = non_stochastic_expr.replace(symbols['f'],
                                                                replace_f)
            # Replace x by the respective variable
            non_stochastic_result = non_stochastic_result.subs(symbols['x'],
                                                               symbols[var])
            # Replace intermediate variables
            temp_vars_specific = dict([('_' + temp_var + '_' + var,
                            sympy.Symbol('_' + temp_var + '_' + var))
                           for temp_var in temp_vars])        
            temp_var_replacements = dict([(temp_var,
                                           temp_vars_specific['_' + temp_var + '_' + var])
                              for temp_var in temp_vars])
            non_stochastic_result = non_stochastic_result.subs(temp_var_replacements)
        else:
            non_stochastic_result = None
        if not (stochastic is None or stochastic_expr is None):
            stochastic_results = []
            
            # We potentially have more than one stochastic variable
            for xi in stochastic:
                # Replace the g(x, t)*xi part
                replace_g = lambda x, t:replace_func(x, t, stochastic[xi],
                                                     temp_vars)
                stochastic_result = stochastic_expr.replace(SYMBOLS['g'],
                                                            replace_g)
                
                # Replace x and xi by the respective variables
                stochastic_result = stochastic_result.subs(SYMBOLS['x'], symbols[var])
                stochastic_result = stochastic_result.subs(SYMBOLS['dW'], xi)   

                # Replace intermediate variables
                temp_vars_specific = dict([('_' + temp_var + '_' + var,
                                sympy.Symbol('_' + temp_var + '_' + var))
                               for temp_var in temp_vars])        
                temp_var_replacements = dict([(temp_var,
                                               temp_vars_specific['_' + temp_var + '_' + var])
                                  for temp_var in temp_vars])
                stochastic_result = stochastic_result.subs(temp_var_replacements)

                stochastic_results.append(stochastic_result)                        
        else:
            stochastic_results = []
        RHS = []
        if non_stochastic_result is not None:
            RHS.append(str(non_stochastic_result))
        for stochastic_result in stochastic_results:
            RHS.append(str(stochastic_result))
        
        RHS = ' + '.join(RHS)
        return RHS

    def __call__(self, eqs):
        '''
        Return "abstract code" for one integration step.
        
        Parameters
        ----------
        eqs : `Equations`
            The model equations that should be integrated.
        
        Returns
        -------
        code : str
            The "abstract code" for the integration step.
        '''

        # The final list of statements
        statements = []
        
        # The variables for the intermedia results in the state updater
        # description, e.g. the variable k in rk2
        intermediate_vars = [var for var, expr in self.statements]
        
        # A dictionary mapping all the variables in the equations to their
        # sympy representations 
        eq_variables = dict([(var, sympy.Symbol(var)) for var in eqs.names])
        
        # The dictionary containing all the symbols used in the state updater
        # description and in the equations
        symbols = self.symbols.copy()
        symbols.update(eq_variables)
        
        # Generate the random numbers for the stochastic variables
        stochastic_variables = eqs.stochastic_variables
        for stochastic_variable in stochastic_variables:
            statements.append(stochastic_variable + ' = ' + 'dt**.5 * randn()')
        
        # Process the intermediate statements in the stateupdater description
        for intermediate_var, intermediate_expr in self.statements:            
            # Split the expression into a non-stochastic and a stochastic part
            non_stochastic_expr, stochastic_expr = split_expression(intermediate_expr)
            # Execute the statement by appropriately replacing the functions f
            # and g and the variable x for every equation in the model
            # (including static equations). 
            for var, expr in eqs.substituted_expressions:
                RHS = self._generate_RHS(eqs, var, symbols, intermediate_vars,
                                         expr, non_stochastic_expr, stochastic_expr)                
                statements.append('_' + intermediate_var + '_' + var + ' = ' + RHS)
                
        # Process the "return" line of the stateupdater description
        non_stochastic_expr, stochastic_expr = split_expression(self.output)
        
        # Assign a value to all the model variables described by differential
        # equations       
        for var, expr in eqs.substituted_expressions:
            RHS = self._generate_RHS(eqs, var, symbols, intermediate_vars,
                                     expr, non_stochastic_expr, stochastic_expr)
            statements.append('_' + var + ' = ' + RHS)
        
        # Assign everything to the final variables
        for var, expr in eqs.substituted_expressions:
            statements.append(var + ' = ' + '_' + var)

        return '\n'.join(statements)

#===============================================================================
# Excplicit state updaters
# Using the arbitrary priority: euler > rk2 > rk4
#===============================================================================

# these objects can be used like functions because they are callable

#: Forward Euler state updater
euler = ExplicitStateUpdater('return x + dt * f(x,t) + g(x,t) * dW',
                             priority=30, stochastic='additive')

#: Second order Runge-Kutta method (midpoint method)
rk2 = ExplicitStateUpdater('''
    k = dt * f(x,t)
    return x + dt*f(x +  k/2, t + dt/2)''', priority=20)

#: Classical Runge-Kutta method (RK4)
rk4 = ExplicitStateUpdater('''
    k1=dt*f(x,t)
    k2=dt*f(x+k1/2,t+dt/2)
    k3=dt*f(x+k2/2,t+dt/2)
    k4=dt*f(x+k3,t+dt)
    return x+(k1+2*k2+2*k3+k4)/6
    ''', priority=10)

#: Derivative-free Milstein method
milstein = ExplicitStateUpdater('''
    x_support = x + dt*f(x, t) + dt**.5 * g(x, t)
    g_support = g(x_support, t)
    k = 1/(2*dt**.5)*(g_support - g(x, t))*(dW**2)
    return x + dt*f(x,t) + g(x, t) * dW + k
    ''', priority=20, stochastic='multiplicative')

# Register the state updaters
StateUpdateMethod.register('euler', euler)
StateUpdateMethod.register('rk2', rk2)
StateUpdateMethod.register('rk4', rk4)
StateUpdateMethod.register('milstein', milstein)