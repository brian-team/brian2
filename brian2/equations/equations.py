# ----------------------------------------------------------------------------------
# Copyright ENS, INRIA, CNRS
# Contributors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
# 
# Brian is a computer program whose purpose is to simulate models
# of biological neural networks.
# 
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# ----------------------------------------------------------------------------------
# 

'''
Differential equations for Brian models.
'''
import keyword
import re
import string

from pyparsing import (Group, ZeroOrMore, OneOrMore, Optional, Word, CharsNotIn,
                       Combine, Suppress, restOfLine, LineEnd, ParseException)

from brian2.units.fundamentalunits import (DimensionMismatchError,
                                           get_dimensions)
from brian2.units.units import second
from brian2.equations.unitcheck import get_unit_from_string
from brian2.equations.codestrings import Expression

# Definitions of equation structure for parsing with pyparsing
###############################################################################
# Basic Elements
###############################################################################

# identifiers like in C: can start with letter or underscore, then a
# combination of letters, numbers and underscores
# Note that the check_identifiers function later performs more checks, e.g.
# names starting with underscore should only be used internally
IDENTIFIER = Word(string.ascii_letters + '_',
                  string.ascii_letters + string.digits + '_').setResultsName('identifier')

# very broad definition here, expression will be analysed by sympy anyway
# allows for multi-line expressions, where each line can have comments
EXPRESSION = Combine(OneOrMore((CharsNotIn(':#\n') +
                                Suppress(Optional(LineEnd()))).ignore('#' + restOfLine)),
                     joinString=' ').setResultsName('expression')


# a unit
# very broad definition here, again. Whether this corresponds to a valid unit
# string will be checked later
UNIT = Word(string.ascii_letters + string.digits + '*/ ').setResultsName('unit')

# a single Flag (e.g. "const" or "event-driven")
FLAG = Word(string.ascii_letters + '_-')

# Flags are comma-separated and enclosed in parantheses: "(flag1, flag2)"
FLAGS = (Suppress('(') + FLAG + ZeroOrMore(Suppress(',') + FLAG) +
         Suppress(')')).setResultsName('flags')

# allowed flags for equation types. Is not used by the parsing directly but
# later for checking
ALLOWED_FLAGS = {'diff_equation': ['active'],
                 'static_equation': [],
                 'parameter': ['constant']}

###############################################################################
# Equations
###############################################################################
# Three types of equations
# Parameter:
# x : volt (flags)
PARAMETER = Group(IDENTIFIER + Suppress(':') + UNIT +
                  Optional(FLAGS)).setResultsName('parameter')

# Static equation:
# x = 2 * y : volt (flags)
STATIC_EQ = Group(IDENTIFIER + Suppress('=') + EXPRESSION + Suppress(':') +
                  UNIT + Optional(FLAGS)).setResultsName('static_equation')

# Differential equation
# dx/dt = -x / tau : volt
DIFFOP = (Suppress('d') + IDENTIFIER + Suppress('/') + Suppress('dt'))
DIFF_EQ = Group(DIFFOP + Suppress('=') + EXPRESSION + Suppress(':') + UNIT +
                Optional(FLAGS)).setResultsName('diff_equation')

# ignore comments
EQUATION = (PARAMETER | STATIC_EQ | DIFF_EQ).ignore('#' + restOfLine)
EQUATIONS = ZeroOrMore(EQUATION)


def check_identifier(identifier, internal=False,
                     reserved_identifiers=('t', 'dt', 'xi')):
    '''
    Check an identifier (usually resulting from an equation string provided by
    the user) for conformity with the rules:
    
        1. Only ASCII characters
        2. Starts with underscore or character, then mix of alphanumerical
           characters and underscore
        3. Is not a reserved keyword of Python
        4. Is not an identifier which has a special meaning for equations, 
           (e.g. "t", "dt", ...)
    
    Arguments:
    
    ``identifier``
        The string that should be checked
    
    ``internal``
        Whether the identifier is defined internally (defaults to ``False``),
        i.e. not by the user. Internal identifiers are allowed to start with an
        underscore whereas user-defined identifiers are not.
    
    ``reserved_identifiers``
        A container of strings, containing identifiers not to be used.
        Defaults to ('t', 'dt', 'xi')
    
    The function raises a ``ValueError`` if the identifier does not conform to
    the above rules.
    '''
    
    # Check whether the identifier is parsed correctly -- this is always the
    # case, if the identifier results from the parsing of an equation but there
    # might be situations where the identifier is specified directly
    parse_result = list(IDENTIFIER.scanString(identifier))
    
    # parse_result[0][0][0] refers to the matched string -- this should be the
    # full identifier, if not it is an illegal identifier like "3foo" which only
    # matched on "foo" 
    if len(parse_result) != 1 or parse_result[0][0][0] != identifier:
        raise ValueError('"%s" is not a valid identifier string.' % identifier)

    if keyword.iskeyword(identifier):
        raise ValueError(('"%s" is a Python keyword and cannot be used as an '
                          'identifier.') % identifier)
    
    if identifier in reserved_identifiers:
        raise ValueError(('"%s" has a special meaning in equations and cannot be '
                         'used as an identifier.') % identifier)
    
    if not internal and identifier.startswith('_'):
        raise ValueError(('Identifier "%s" starts with an underscore, '
                          'this is only allowed for variables used internally') % identifier)


class Equations(object):
    """Container that stores equations from which models can be created
    
    Initialised as::
    
        Equations(expr[,level=0[,keywords...]])
    
    with arguments:
    
    ``eqns``
        String equation(s) (possibly multi-line)
    ``allowed_flags``:
        A dictionary with a list of allowed flags (strings) for the equation
        types ``diff_equation``, ``static_equation`` and ``parameter``. Not
        defining allowed flags for a key corresponds to setting it to an empty
        list. This is used e.g. by the :class:``Synapses`` class, allowing
        "event-driven" for differential equations. If ``allowed_flags`` is
        ``None`` (the default), standard settings are used: ``active`` is
        allowed for differential equations, ``constant`` is used for parameters.
    ``reserved_identifiers``:
        A list or set of strings that are not allowed as identifiers. Will be
        added to ['t', 'dt', 'xi'] which are always forbidden. This is used by
        the :class:``Synapses`` class which forbids "i" and "j" as identifiers.        
    ``namespace``:
        TODO
    
    **String equations**
    
    String equations can be of any of the following forms:
    
    (1) ``dx/dt = f : unit (flags)`` (differential equation)
    (2) ``x = f : unit (flags)`` (equation)
    (3) ``x : unit (flags)`` (parameter)
    
    """
    
    class _Equation():
        '''
        Class for internal use, encapsulates a single equation or parameter.
        '''
        def __init__(self, eq_type, varname, expr, unit, flags,
                     namespace, exhaustive, level):
            '''
            Create a new :class:`_Equation` object.
            '''
            self.eq_type = eq_type
            self.varname = varname
            if eq_type != 'parameter':
                self.expr = Expression(expr, namespace=namespace,
                                       exhaustive=exhaustive, level=level + 1)
            else:
                self.expr = None
            self.unit = unit
            self.flags = flags

        def resolve(self, internal_variables):
            if not self.expr is None:
                self.expr.resolve(internal_variables)

        def __str__(self):
            if self.eq_type == 'diff_equation':
                s = 'd' + self.varname + '/dt'
            else:
                s = self.varname
            
            if self.eq_type in ['diff_equation', 'static_equation']:
                s += ' = ' + str(self.expr)
            
            s += ' : ' + str(self.unit)
            
            if len(self.flags):
                s += '(' + ', '.join(self.flags) + ')'
            
            return s

    
    def __init__(self, eqns='', namespace=None, exhaustive=False, level=0):
                
        self._equations = self._parse_string_equations(eqns, namespace,
                                                       exhaustive, level)
        
        # For convenience, a dictionary with all the units
        self._units = dict([(var, eq.unit) for var, eq in
                            self._equations.iteritems()])
        # Add the special variables to the unit dicitionary
        self._units.update({'t': second, 'dt': second, 'xi': second**0.5})
        
        # A set of all the state variables / parameters
        self._variables = set(self._units.keys())
        
        # Build the namespaces, resolve all external variables and rearrange
        # static equations
        self._resolve()
        
        # Check the units for consistency
        self.check_units()
        
        # TODO: Separate stochastic and non-stochastic parts
        

    def _parse_string_equations(self, eqns, namespace, exhaustive, level):
        """
        Parses a string defining equations and returns a dictionary, mapping
        variable names to :class:`Equations._Equation` objects.
        """
        equations = {}
        
        try:
            parsed = EQUATIONS.parseString(eqns, parseAll=True)
        except ParseException as p_exc:
            # TODO: Any way to have colorful output when run under ipython?
            raise ValueError('Parsing failed: \n' + str(p_exc.line) + '\n' +
                             ' '*(p_exc.column - 1) + '^\n' + str(p_exc))
        for eq in parsed:
            eq_type = eq.getName()
            eq_content = dict(eq.items())
            # Check for reserved keywords
            identifier = eq_content['identifier']
            check_identifier(identifier)
            
            # Convert unit string to Unit object
            unit = get_unit_from_string(eq_content['unit'])
            
            expression = eq_content.get('expression', None)
            if not expression is None:
                # Replace multiple whitespaces (arising from joining multiline
                # strings) with single space
                p = re.compile(r'\s{2,}')
                expression = p.sub(' ', expression)
            flags = eq_content.get('flags', [])
    
            equation = Equations._Equation(eq_type, identifier, expression,
                                           unit, flags, namespace,
                                           exhaustive, level + 1) 
            
            if identifier in equations:
                raise ValueError('Duplicate definition of variable "%s"' %
                                 identifier)
                                           
            equations[identifier] = equation
        
        return equations            

    def _resolve(self):
        for eq in self._equations.itervalues():
            eq.resolve(self._variables)
        
        #TODO: Build a single namespace for the equations object
        #TODO: Check for dependencies and reorder static equations

    def check_units(self):
        for var, eq in self._equations.iteritems():
            if eq.eq_type == 'parameter':
                # no need to check units for parameters
                continue
            
            if eq.eq_type == 'diff_equation':
                try:
                    eq.expr.check_unit_against(self._units[var] / second,
                                               self._units)
                except DimensionMismatchError as dme:
                    raise DimensionMismatchError(('Differential equation defining '
                                                  '%s does not use consistent units: %s') % 
                                                 (var, dme.desc), *dme._dims)
            elif eq.eq_type == 'static_equation':
                try:
                    eq.expr.check_unit_against(self._units[var],
                                               self._units)
                except DimensionMismatchError as dme:
                    raise DimensionMismatchError(('Static equation defining '
                                                  '%s does not use consistent units: %s') % 
                                                 (var, dme.desc), *dme._dims)                
            else:
                raise AssertionError('Unknown equation type: "%s"' % eq.eq_type)
        
    #
    # Representation
    # 

    def __str__(self):
        strings = [str(eq) for eq in self._equations.itervalues()]
        return '\n'.join(strings)

    def _repr_pretty_(self, p, cycle):
        ''' Pretty printing for ipython '''
        if cycle: 
            # Should never happen actually
            return 'Equations(...)'
        for eq in self._equations.itervalues():
            p.pretty(eq)            

if __name__ == '__main__':
    tau = 5 * second
    eqs = Equations('''dv/dt = -(v + x)/ tau : volt
                       x : 1
                       y = 2 * v : volt''')
    