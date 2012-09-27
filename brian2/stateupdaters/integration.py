import string
from sympy import sympify, Symbol, Function
from pyparsing import (Literal, Group, Word, ZeroOrMore, Suppress, restOfLine,
                       ParseException)

__all__ = ['euler', 'rk2', 'rk4', 'ExplicitStateUpdater']

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
SYMBOLS = {'x' : Symbol('x'),
           't' : Symbol('t'),
           'dt': Symbol('dt'),
           'f' : Function('f')}

class ExplicitStateUpdater(object):
    '''
    An object that can be used for defining state updaters via a simple
    description (see :meth:`__init__` for details). Objects can be passed to
    the ``method`` argument of the :class:`NeuronGroup` constructor. As other
    state updater functions the :class:`ExplicitStateUpdater` objects are 
    callable, returning abstract code when called with an :class:`Equations`
    object.
    ''' 
    
    def __init__(self, description):
        '''
        Create a new state updater from the given ``description``.
        
        Example for such a description string (defining Runge-Kutta 4):
            k1 = dt*f(x,t)
            k2 = dt*f(x+k1/2,t+dt/2)
            k3 = dt*f(x+k2/2,t+dt/2)
            k4 = dt*f(x+k3,t+dt)
            return x+(k1+2*k2+2*k3+k4)/6
        '''
        
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
            expression = sympify(element.expression, locals=self.symbols)
            symbols = list(expression.atoms(Symbol))
            self.symbols.update(dict([(symbol.name, symbol)
                                      for symbol in symbols]))
            if element.getName() == 'statement':
                self.statements.append((element.identifier, expression))
            elif element.getName() == 'output':
                self.output = expression
            else:
                raise AssertionError('Unknown element name: %s' % element.getName())
    
    def __str__(self):
        s = ''
        
        if len(self.statements) > 0:
            s += 'Intermediate statements:\n'
            s += '\n'.join([(var + ' = ' + str(expr)) for var, expr in self.statements])
            s += '\n'
            
        s += 'Output:\n'
        s += self.output
        return s
    
    def __call__(self, eqs):
        '''
        Return "abstract code" for the given :class:`Equations` object ``eqs``.
        '''
                
        def replace_func(x, t, expr, temp_vars):
            s_expr = sympify(expr, locals=self.symbols)
            
            for var in eqs.eq_names:
                temp_vars_specific = dict([('_' + temp_var + '_' + var,
                                            Symbol('_' + temp_var + '_' + var))
                                           for temp_var in temp_vars])
                self.symbols.update(temp_vars_specific)
                temp_var_replacements = dict([(temp_var, temp_vars_specific['_' + temp_var + '_' + var])
                                              for temp_var in temp_vars])
                one_replacement = x.subs(SYMBOLS['x'], variables[var])
                                
                one_replacement = one_replacement.subs(temp_var_replacements)
                
                s_expr = s_expr.subs(variables[var], one_replacement)
            
            # replace time (important for time-dependent equations)
            s_expr.subs(SYMBOLS['t'], t)
            return s_expr

        statements = []
        temp_vars = [var for var, expr in self.statements]
        variables = dict([(var, Symbol(var)) for var in eqs.names])
        self.symbols.update(variables)
        # Intermediate statements
        for temp_var, temp_expr in self.statements:
            for var, expr in eqs.eq_expressions:                                

                temp_result = temp_expr.replace(SYMBOLS['f'],
                                                lambda x, t: replace_func(x, t, expr, temp_vars))                
                statements.append('_' + temp_var + '_' + var + ' = ' + str(temp_result))
                
        # The "return" line        
        for var, expr in eqs.diff_eq_expressions:
            # Handle f(x, t) calls                                
            temp_result = self.output.replace(SYMBOLS['f'],
                                              lambda x, t: replace_func(x, t, expr, temp_vars))
            # Handle references to variables and intermediate variables
            temp_result = temp_result.subs(SYMBOLS['x'], variables[var])
            for temp_var in temp_vars:
                temp_result = temp_result.subs(self.symbols[temp_var],
                                               self.symbols['_' + temp_var + '_' + var])
            statements.append('_' + var + ' = ' + str(temp_result))
        
        # Assign everything to the final variables
        for var, expr in eqs.diff_eq_expressions:
            statements.append(var + ' = ' + '_' + var)

        return '\n'.join(statements)

#===============================================================================
# Excplicit state updaters
#===============================================================================

# these objects can be used like functions because they are callable     
euler = ExplicitStateUpdater('return x + dt * f(x,t)')


rk2 = ExplicitStateUpdater('''
    k = dt * f(x,t)
    return x + dt*f(x +  k/2, t + dt/2)''')


rk4 = ExplicitStateUpdater('''
    k1=dt*f(x,t)
    k2=dt*f(x+k1/2,t+dt/2)
    k3=dt*f(x+k2/2,t+dt/2)
    k4=dt*f(x+k3,t+dt)
    return x+(k1+2*k2+2*k3+k4)/6
    ''')
