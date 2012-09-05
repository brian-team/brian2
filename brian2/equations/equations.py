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
import inspect
import keyword
import re
import string
import uuid

from pyparsing import (Group, ZeroOrMore, OneOrMore, Optional, Word, CharsNotIn,
                       Combine, Suppress, restOfLine, LineEnd, ParseException)
from sympy import sympify, Symbol, Wild
from sympy.core.sympify import SympifyError

from brian2.units.stdunits import stdunits
from brian2.units.fundamentalunits import (Quantity, Unit, all_registered_units,
                                           have_same_dimensions, DIMENSIONLESS,
                                           DimensionMismatchError,
                                           get_dimensions)
from brian2.units import second

from brian.inspection import get_identifiers

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


def unique_id():
    """
    Returns a unique name (e.g. for internal hidden variables).
    """
    return '_' + str(uuid.uuid1().int) 


def get_default_unit_namespace():    
    namespace = dict([(u.name, u) for u in all_registered_units()])
    namespace.update(stdunits)
    return namespace


def get_expression_parts(expr):
    '''
    Returns a tuple containing the functions f and g (as sympy expressions),
    assuming expressions of the form ``f + g * xi``, where ``xi`` is the
    symbol for the random variable.
    
    ``expr`` can be a string or a sympy expression.
    
    Always returns expressions for f and g, but these can be ``0`` if the
    corresponding part is not present. 
    
    Examples::
    
        >>> get_expression_parts('-v / (1 * ms)')
        (-v/ms, 0)
        >>> get_expression_parts('sigma * xi')
        (0, sigma)
        >>> get_expression_parts('mu / tau + sigma / tau**.5 * xi')
        (mu/tau, sigma*tau**-0.5)
        >>> get_expression_parts('-v + xi ** 2')
        Traceback (most recent call last):
            ...    
        ValueError: Expression "-v + xi**2" cannot be separated into stochastic and non-stochastic term
    '''
    expr = sympify(expr)
    xi = Symbol('xi')
    if not xi in expr:
        return (expr, sympify('0'))
    
    f = Wild('f', exclude=[xi]) # non-stochastic part
    g = Wild('g', exclude=[xi]) # stochastic part
    matches = expr.match(f + g * xi)
    if matches is None:
        raise ValueError(('Expression "%s" cannot be separated into stochastic '
                         'and non-stochastic term') % expr)

    return (matches[f], matches[g])


def get_unit_from_string(unit_string, unit_namespace=None):
    if unit_namespace is None:
        namespace = get_default_unit_namespace()
    else:
        namespace = unit_namespace
    unit_string = unit_string.strip()
    
    # Special case: dimensionless unit
    if unit_string == '1':
        return Unit(1, dim=DIMENSIONLESS)
    
    # Check first whether the expression evaluates at all, using only
    # registered units
    try:
        evaluated_unit = eval(unit_string, namespace)
    except Exception as ex:
        raise ValueError('"%s" does not evaluate to a unit: %s' %
                         (unit_string, ex))
    
    # Check whether the result is a unit
    if not isinstance(evaluated_unit, Unit):
        if isinstance(evaluated_unit, Quantity):
            raise ValueError(('"%s" does not evaluate to a unit but to a '
                              'quantity -- make sure to only use units, e.g. '
                              '"siemens/m**2" and not "1 * siemens/m**2"') %
                             unit_string)
        else:
            raise ValueError(('"%s" does not evaluate to a unit, the result '
                             'has type %s instead.' % (unit_string,
                                                       type(evaluated_unit))))
    # We only want base units, otherwise e.g. setting a unit to mV might lead to 
    # unexpected results (as it is internally saved in volts)
    # TODO: Maybe this restriction is unnecessary with unit arrays?
    if float(evaluated_unit) != 1.0:
        raise ValueError(('"%s" is not a base unit, but only base units are '
                         'allowed in the units part of equations.') % unit_string)

    # No error has been raised, all good
    return evaluated_unit

def get_namespace_for_codestring(codestring, other_identifiers, local_namespace,
                                 global_namespace):
    '''
    Returns a namespace for the given codestring, containing resolved
    references to externally defined variables and functions. Raises an error
    if a variable/function cannot be resolved.
    
    The resulting namespace includes units but does not include anything
    present in the ``other_identifiers`` set.
    '''
    unit_namespace = get_default_unit_namespace()

    identifiers = set(get_identifiers(codestring))
    namespace = {}
    for identifier in identifiers:
        if identifier in other_identifiers:
            pass
        elif identifier in local_namespace:
            namespace[identifier] = local_namespace[identifier]
        elif identifier in global_namespace:
            namespace[identifier] = global_namespace[identifier]
        elif identifier in unit_namespace:
            namespace[identifier] = unit_namespace[identifier]
        else:
            raise ValueError(('The string "%s" refers to an unknown '
                              'identifier "%s"') % (codestring, identifier))    
    return namespace


def get_dimensions_for_codestring(codestring, identifier_units, namespace):
    '''
    Returns the dimensions of the ``codestring`` by evaluating it in the given
    ``namespace``, replacing all identifiers with their units. The units have
    to be given in the dictionary ``identifier_units``.
    
    May raise an DimensionMismatchError during the evaluation.
    '''
    namespace = namespace.copy()
    namespace.update(identifier_units)
    return get_dimensions(eval(codestring, namespace))


def get_frozen_codestring_and_namespace(codestring, namespace):
    '''
    Returns a tuple containing a codestring and a namespace, resulting from
    replacing everything in the given ``codestring`` from ``namespace`` by its
    float value. Removes the corresponding entries from the namespace.
    '''
    new_namespace = {}    
    #TODO: Use sympy instead to simplify the expression?
    for key, value in namespace.iteritems():
        try:
            float_value = float(value)
            codestring = re.sub(r'\b' + key + r'\b', str(float_value),
                                codestring)            
        except (TypeError, ValueError):
            # could not convert the identifier to a float, keep it in the
            # namespace
            new_namespace[key] = value
    
    return (codestring, new_namespace)


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
    def __init__(self, eqns='', level=0, allowed_flags=None, 
                 reserved_identifiers=None, namespace=None):
        # Empty object        
        self._string_expressions = {} # dictionary of strings (defining the functions)
        self._sympy_expressions = {} # dictionary of sympy expressions
        self._sympy_expressions_non_stochastic = {} # non-stochastic parts of diff equations
        self._sympy_expressions_stochastic = {} # stochastic parts of diff equations
        self._namespaces = {} # dictionary of namespaces for the strings        
        self._units = {'t':second, 'dt': second, 'xi': second ** -.5} # dictionary of units
        self._flags = {} # dictionary of "flags", e.g. "const" or "event-driven"
        self._identifiers = set(['t', 'dt', 'xi']) # all identifiers (union of the next 3 lists)
        self._diff_equation_names = [] # differential equation variables
        self._static_equation_names = [] # static equation variables
        self._parameter_names = [] # parameters
        self._dependencies = {} # dictionary of dependencies (on static equations)
        
        if allowed_flags is None:
            self.allowed_flags = ALLOWED_FLAGS
        else:
            self.allowed_flags = allowed_flags

        self.reserved_identifiers = set(['t', 'dt', 'xi'])
        if not reserved_identifiers is None:
            self.reserved_identifiers = self.reserved_identifiers.union(set(reserved_identifiers))
        
        self.parse_string_equations(eqns)

        if namespace is None:
            frame = inspect.stack()[level + 1][0]
            ns_global, ns_local = frame.f_globals, frame.f_locals
        else:
            ns_local = namespace
            ns_global = {}

        # build the namespaces
        for var in self._static_equation_names + self._diff_equation_names:
            self._namespaces[var] = get_namespace_for_codestring(self._string_expressions[var],
                                                                 self._identifiers,
                                                                 ns_local,
                                                                 ns_global)
        
        # check the units
        self.check_units()

    """
    -----------------------------------------------------------------------
    PARSING AND BUILDING NAMESPACES
    -----------------------------------------------------------------------
    """

    def parse_string_equations(self, eqns):
        """
        Parses a string defining equations and builds an Equations object.
        Uses the namespace in the given level of the stack.
        """
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
            self.check_identifier(identifier)
            
            # Convert unit string to Unit object
            unit = get_unit_from_string(eq_content['unit'])
            
            expression = eq_content.get('expression', None)
            if not expression is None:
                # Replace multiple whitespaces (arising from joining multiline
                # strings) with single space
                p = re.compile(r'\s{2,}')
                expression = p.sub(' ', expression)
            flags = eq_content.get('flags', None)
            # TODO: Take care of namespaces
            self.add_equation(identifier, eq_type, expression, unit, flags)

    def add_equation(self, identifier, eq_type, eq, unit, flags):
        """
        Inserts a differential equation.
        name = variable name
        eq = string definition
        unit = unit of the variable (possibly a string)
        *_namespace = namespaces associated to the string        
        """
        if identifier in self._identifiers:
            raise ValueError('Duplicate definition of "%s".' % identifier)
        else:
            self._identifiers.add(identifier)
        
        self._units[identifier] = unit

        self._string_expressions[identifier] = eq
        if eq is not None:
            try:
                self._sympy_expressions[identifier] = sympify(eq)
            except SympifyError as s_exc:
                raise ValueError('Sympy parsing failed for equation definining %s:\n%s' %
                                 (identifier, s_exc))
        if eq_type == 'diff_equation':
            self._diff_equation_names.append(identifier)
            # get stochastic and non-stochastic parts of equation
            (f, g) = get_expression_parts(self._sympy_expressions[identifier])
            self._sympy_expressions_non_stochastic[identifier] = f
            self._sympy_expressions_stochastic[identifier] = g
        elif eq_type == 'static_equation':
            self._static_equation_names.append(identifier)
        elif eq_type == 'parameter':
            self._parameter_names.append(identifier)
        else:
            raise ValueError('Unknown equation type "%s"' % eq_type)
        
        if flags is None:
            self._flags[identifier] = []
        else:
            self.check_flags(flags, eq_type)
            self._flags[identifier] = flags

    def check_identifier(self, identifier, internal=False):
        '''
        Check an identifier (usually resulting from an equation string provided by
        the user) for conformity with the rules:
        
            1. Only ASCII characters
            2. Starts with underscore or character, then mix of alphanumerical
               characters and underscore
            3. Is not a reserved keyword of Python
            4. Is not an identifier which has a special meaning for equations 
               (e.g. "t", "dt", ...)
        
        Arguments:
        
        ``identifier``
            The string that should be checked
        
        ``internal``
            Whether the identifier is defined internally (defaults to ``False``),
            i.e. not by the user. Internal identifiers are allowed to start with an
            underscore whereas user-defined identifiers are not.
        
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
            raise ValueError('"%s" is a Python keyword and cannot be used as an identifier.' % identifier)
        
        if identifier in self.reserved_identifiers:
            raise ValueError(('"%s" has a special meaning in equations and cannot be '
                             'used as an identifier.') % identifier)
        
        if not internal and identifier.startswith('_'):
            raise ValueError(('Identifier "%s" starts with an underscore, '
                              'this is only allowed for variables used internally') % identifier)            

    def check_flags(self, flags, eq_type):        
        for flag in flags:
            allowed = self.allowed_flags.get(eq_type, [])
            if not flag in allowed:
                raise ValueError(('Flag "%s" is not allowed for equations of '
                                 'type "%s". Allowed flags are: %s') % 
                                 (flag, eq_type, allowed))

    def _get_units_for_RHS(self, var):
        try:
            dim_RHS = get_dimensions_for_codestring(self._string_expressions[var],
                                                    self._units, self._namespaces[var])                
        except DimensionMismatchError as dme:
            raise DimensionMismatchError(('Error during unit check of expression '
                                          'defining %s: %s') % (var, dme.desc),
                                         *dme._dims)
        return dim_RHS

    def check_units(self):
        for var in self._diff_equation_names:            
            dim_RHS = self._get_units_for_RHS(var)
            dim_LHS = get_dimensions(self._units[var] / second)
            if not dim_LHS is dim_RHS:
                raise DimensionMismatchError(('Differential equation defining '
                                              '%s is not homogeneous') % var,
                                             dim_LHS, dim_RHS)
        for var in self._static_equation_names:
            dim_RHS = self._get_units_for_RHS(var)            
            dim_LHS = get_dimensions(self._units[var])
            if not dim_LHS is dim_RHS:
                raise DimensionMismatchError(('Static equation defining '
                                              '%s is not homogeneous') % var,
                                             dim_LHS, dim_RHS)

        
    #
    # Representation
    # 

    def __str__(self):
        s = ''
        for var in self._diff_equation_names:
            s += ('d' + var + '/dt = ' + self._string_expressions[var] + ': ' +
                  str(self._units[var]))
            if len(self._flags[var]):
                s += ' (' + ' ,'.join(self._flags[var]) + ')'
            s += '\n'
        for var in self._static_equation_names:
            s += (var + ' = ' + self._string_expressions[var] + ': ' +
                  str(self._units[var]))
            if len(self._flags[var]):
                s += ' (' + ', '.join(self._flags[var]) + ')'            
            s += '\n'
        for var in self._parameter_names:
            s += (var + ': ' + str(self._units[var]))
            if len(self._flags[var]):
                s += ' (' + ', '.join(self._flags[var]) + ')'            
            s += '\n'
        return s

    def _repr_pretty_(self, p, cycle):
        ''' Pretty printing for ipython '''
        if cycle: 
            # Should never happen actually
            return 'Equations(...)'
        for var in self._diff_equation_names:
            with p.group(len(var) + 7, 'd' + var + '/dt = ', ''):
                p.pretty(self._sympy_expressions[var])
                p.text(' :')
                p.breakable()
                p.pretty(self._units[var])
                if len(self._flags[var]):
                    p.text(' (')
                    p.text(self._flags[var].join(', '))
                    p.text(' )')
                p.text('\n')
        for var in self._static_equation_names:
            with p.group(len(var) + 3,  var + ' = ', ''):
                p.pretty(self._sympy_expressions[var])
                p.text(' :')
                p.breakable()
                p.pretty(self._units[var])
                if len(self._flags[var]):
                    p.text(' (')
                    p.text(self._flags[var].join(', '))
                    p.text(' )')                
                p.text('\n')
        for var in self._parameter_names:
                p.text(var)
                p.text(' :')
                p.breakable()
                p.pretty(self._units[var])
                if len(self._flags[var]):
                    p.text(' (')
                    p.text(self._flags[var].join(', '))
                    p.text(' )')                
                p.text('\n')

if __name__ == '__main__':
    import doctest