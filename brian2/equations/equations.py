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
import string
import keyword
import inspect

import sympy
from pyparsing import (Group, ZeroOrMore, OneOrMore, Optional, Word, CharsNotIn,
                       Combine, Suppress, restOfLine, LineEnd)

from brian2.units.fundamentalunits import Quantity, Unit, all_registered_units

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
UNIT = EXPRESSION.setResultsName('unit')

# a single Flag (e.g. "const" or "event-driven")
FLAG = Word(string.ascii_letters + '_-')

# Flags are comma-separated and enclosed in parantheses: "(flag1, flag2)"
FLAGS = (Suppress('(') + FLAG + ZeroOrMore(Suppress(',') + FLAG) +
         Suppress(')')).setResultsName('flags')

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


def is_reserved_identifier(identifier):
    # TODO: Should "i" and "j" always be forbidden or only in Synapses code?
    return identifier in ['t', 'dt', 'xi', 'i', 'j'] 


def check_identifier(identifier, internal=False):
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
    
    if is_reserved_identifier(identifier):
        raise ValueError(('"%s" has a special meaning in equations and cannot be '
                         'used as an identifier.') % identifier)
    
    if not internal and identifier.startswith('_'):
        raise ValueError(('Identifier "%s" starts with an underscore, '
                          'this is only allowed for variables used internally') % identifier)
    

def get_unit_from_string(unit_string):
    unit_string = unit_string.trim()
    
    # Special case: dimensionless unit
    if unit_string == '1':
        return Unit(1)
    
    # Check first whether the expression evaluates at all, using only
    # registered units
    try:
        namespace = dict([(u.name, u) for u in all_registered_units()])
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


class Equations(object):
    """Container that stores equations from which models can be created
    
    Initialised as::
    
        Equations(expr[,level=0[,keywords...]])
    
    with arguments:
    
    ``expr``
        A string equation
    ``keywords``
        Any sequence of keyword pairs ``key=value`` where the string ``key``
        in the string equations will be replaced with ``value`` which can
        be either a string, value or ``None``, in the latter case a unique
        name will be generated automatically (but it won't be pretty).
    
    **String equations**
    
    String equations can be of any of the following forms:
    
    (1) ``dx/dt = f : unit (flags)`` (differential equation)
    (2) ``x = f : unit (flags)`` (equation)
    (4) ``x : unit (flags)`` (parameter)
    
    """
    def __init__(self, expr='', level=0, **kwds):
        # Empty object
        self._Vm = None # name of variable with membrane potential
        self._eq_names = [] # equations names
        self._diffeq_names = [] # differential equations names
        self._diffeq_names_nonzero = [] # differential equations names
        self._function = {} # dictionary of functions
        self._string = {} # dictionary of strings (defining the functions)
        self._namespace = {} # dictionary of namespaces for the strings (globals,locals)
        self._alias = {} # aliases (mapping name1 -> name2)
        self._units = {'t':second} # dictionary of units
        self._dependencies = {} # dictionary of dependencies (on static equations)

        self._frozen = False # True if all units and parameters are gone
        self._prepared = False

        if not isinstance(expr, str): # assume it is a sequence of Equations objects
            for eqs in expr:
                if not isinstance(eqs, Equations):
                    eqs = Equations(eqs, level=level + 1)
                self += eqs
        elif expr != '':
            # Check keyword arguments
            param_dict = {}
            for name, value in kwds.iteritems():
                if value is None: # name is not important: choose unique name
                    value = unique_id()
                if isinstance(value, str): # variable name substitution
                    expr = re.sub('\\b' + name + '\\b', value, expr)
                    expr = re.sub('\\bd' + name + '\\b', 'd' + value, expr) # derivative
                else:
                    param_dict[name] = value

        self.parse_string_equations(expr, namespace=param_dict, level=level + 1)

    """
    -----------------------------------------------------------------------
    PARSING AND BUILDING NAMESPACES
    -----------------------------------------------------------------------
    """

    def parse_string_equations(self, eqns, level=1, namespace=None):
        """
        Parses a string defining equations and builds an Equations object.
        Uses the namespace in the given level of the stack.
        """
        parsed = EQUATIONS.parseString(eqns, parseAll=True)
        for eq in parsed:
            eq_type = eq.getName()
            eq_content = dict(eq.get_items())
            # Check for reserved keywords
            check_identifier(eq_content['identifier'])
            
            # Convert unit string to Unit object
            unit = get_unit_from_string(eq_content['unit'])

