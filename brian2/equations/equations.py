'''
Differential equations for Brian models.
'''
import collections
import keyword
import re
import string
import numpy as np
import sympy
from pyparsing import (Group, ZeroOrMore, OneOrMore, Optional, Word, CharsNotIn,
                       Combine, Suppress, restOfLine, LineEnd, ParseException)

from brian2.core.namespace import (DEFAULT_FUNCTIONS,
                                   DEFAULT_CONSTANTS,
                                   DEFAULT_UNITS)
from brian2.core.variables import Constant
from brian2.core.functions import Function
from brian2.parsing.sympytools import sympy_to_str, str_to_sympy
from brian2.units.fundamentalunits import (Unit, Quantity, have_same_dimensions,
                                           get_unit, DIMENSIONLESS,
                                           DimensionMismatchError)
from brian2.units.allunits import (metre, meter, second, amp, kelvin, mole,
                                   candle, kilogram, radian, steradian, hertz,
                                   newton, pascal, joule, watt, coulomb, volt,
                                   farad, ohm, siemens, weber, tesla, henry,
                                   celsius, lumen, lux, becquerel, gray,
                                   sievert, katal, kgram, kgramme)
from brian2.utils.logger import get_logger
from brian2.utils.topsort import topsort

from .codestrings import Expression
from .unitcheck import check_unit


__all__ = ['Equations']

logger = get_logger(__name__)

# Equation types (currently simple strings but always use the constants,
# this might get refactored into objects, for example)
PARAMETER = 'parameter'
DIFFERENTIAL_EQUATION = 'differential equation'
SUBEXPRESSION = 'subexpression'

# variable types (FLOAT is the only one that is possible for variables that
# have dimensions). These types will be later translated into dtypes, either
# using the default values from the preferences, or explicitly given dtypes in
# the construction of the `NeuronGroup`, `Synapses`, etc. object
FLOAT = 'float'
INTEGER = 'integer'
BOOLEAN = 'boolean'

# Definitions of equation structure for parsing with pyparsing
# TODO: Maybe move them somewhere else to not pollute the namespace here?
#       Only IDENTIFIER and EQUATIONS are ever used later
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
UNIT = Word(string.ascii_letters + string.digits + '*/.- ').setResultsName('unit')

# a single Flag (e.g. "const" or "event-driven")
FLAG = Word(string.ascii_letters + '_- ')

# Flags are comma-separated and enclosed in parantheses: "(flag1, flag2)"
FLAGS = (Suppress('(') + FLAG + ZeroOrMore(Suppress(',') + FLAG) +
         Suppress(')')).setResultsName('flags')

###############################################################################
# Equations
###############################################################################
# Three types of equations
# Parameter:
# x : volt (flags)
PARAMETER_EQ = Group(IDENTIFIER + Suppress(':') + UNIT +
                     Optional(FLAGS)).setResultsName(PARAMETER)

# Static equation:
# x = 2 * y : volt (flags)
STATIC_EQ = Group(IDENTIFIER + Suppress('=') + EXPRESSION + Suppress(':') +
                  UNIT + Optional(FLAGS)).setResultsName(SUBEXPRESSION)

# Differential equation
# dx/dt = -x / tau : volt
DIFF_OP = (Suppress('d') + IDENTIFIER + Suppress('/') + Suppress('dt'))
DIFF_EQ = Group(DIFF_OP + Suppress('=') + EXPRESSION + Suppress(':') + UNIT +
                Optional(FLAGS)).setResultsName(DIFFERENTIAL_EQUATION)

# ignore comments
EQUATION = (PARAMETER_EQ | STATIC_EQ | DIFF_EQ).ignore('#' + restOfLine)
EQUATIONS = ZeroOrMore(EQUATION)


class EquationError(Exception):
    '''
    Exception type related to errors in an equation definition.
    '''
    pass


def check_identifier_basic(identifier):
    '''
    Check an identifier (usually resulting from an equation string provided by
    the user) for conformity with the rules. The rules are:
    
        1. Only ASCII characters
        2. Starts with a character, then mix of alphanumerical characters and
           underscore
        3. Is not a reserved keyword of Python
    
    Parameters
    ----------    
    identifier : str
        The identifier that should be checked
    
    Raises
    ------
    ValueError    
        If the identifier does not conform to the above rules.
    '''

    # Check whether the identifier is parsed correctly -- this is always the
    # case, if the identifier results from the parsing of an equation but there
    # might be situations where the identifier is specified directly
    parse_result = list(IDENTIFIER.scanString(identifier))

    # parse_result[0][0][0] refers to the matched string -- this should be the
    # full identifier, if not it is an illegal identifier like "3foo" which only
    # matched on "foo"
    if len(parse_result) != 1 or parse_result[0][0][0] != identifier:
        raise ValueError('"%s" is not a valid variable name.' % identifier)

    if keyword.iskeyword(identifier):
        raise ValueError(('"%s" is a Python keyword and cannot be used as a '
                          'variable.') % identifier)

    if identifier.startswith('_'):
        raise ValueError(('Variable "%s" starts with an underscore, '
                          'this is only allowed for variables used '
                          'internally') % identifier)


def check_identifier_reserved(identifier):
    '''
    Check that an identifier is not using a reserved special variable name. The
    special variables are: 't', 'dt', and 'xi', as well as everything starting
    with `xi_`.
    
    Parameters
    ----------
    identifier: str
        The identifier that should be checked
    
    Raises
    ------
    ValueError
        If the identifier is a special variable name.
    '''
    if identifier in ('t', 'dt', 'xi') or identifier.startswith('xi_'):
        raise ValueError(('"%s" has a special meaning in equations and cannot '
                         ' be used as a variable name.') % identifier)


def check_identifier_units(identifier):
    '''
    Make sure that identifier names do not clash with unit names.
    '''
    if identifier in DEFAULT_UNITS:
        raise ValueError('"%s" is the name of a unit, cannot be used as a '
                         'variable name.' % identifier)

def check_identifier_functions(identifier):
    '''
    Make sure that identifier names do not clash with function names.
    '''
    if identifier in DEFAULT_FUNCTIONS:
        raise ValueError('"%s" is the name of a function, cannot be used as a '
                         'variable name.' % identifier)

def check_identifier_constants(identifier):
    '''
    Make sure that identifier names do not clash with function names.
    '''
    if identifier in DEFAULT_CONSTANTS:
        raise ValueError('"%s" is the name of a constant, cannot be used as a '
                         'variable name.' % identifier)


def unit_and_type_from_string(unit_string):
    '''
    Returns the unit that results from evaluating a string like
    "siemens / metre ** 2", allowing for the special string "1" to signify
    dimensionless units, the string "boolean" for a boolean and "integer" for
    an integer variable.

    Parameters
    ----------
    unit_string : str
        The string that should evaluate to a unit

    Returns
    -------
    u, type : (Unit, {FLOAT, INTEGER or BOOL})
        The resulting unit and the type of the variable.

    Raises
    ------
    ValueError
        If the string cannot be evaluated to a unit.
    '''

    # We avoid using DEFAULT_NUMPY_NAMESPACE here as importing core.namespace
    # would introduce a circular dependency between it and the equations
    # package
    base_units = [metre, meter, second, amp, kelvin, mole, candle, kilogram,
                  radian, steradian, hertz, newton, pascal, joule, watt,
                  coulomb, volt, farad, ohm, siemens, weber, tesla, henry,
                  celsius, lumen, lux, becquerel, gray, sievert, katal, kgram,
                  kgramme]
    namespace = dict((repr(unit), unit) for unit in base_units)
    namespace['Hz'] = hertz  # Also allow Hz instead of hertz
    unit_string = unit_string.strip()

    # Special case: dimensionless unit
    if unit_string == '1':
        return Unit(1, dim=DIMENSIONLESS), FLOAT

    # Another special case: boolean variable
    if unit_string == 'boolean':
        return Unit(1, dim=DIMENSIONLESS), BOOLEAN
    if unit_string == 'bool':
        raise TypeError("Use 'boolean' not 'bool' as the unit for a boolean "
                        "variable.")

    # Yet another special case: integer variable
    if unit_string == 'integer':
        return Unit(1, dim=DIMENSIONLESS), INTEGER

    # Check first whether the expression evaluates at all, using only base units
    try:
        evaluated_unit = eval(unit_string, namespace)
    except Exception as ex:
        raise ValueError(('"%s" does not evaluate to a unit when only using '
                          'base units (e.g. volt but not mV): %s') %
                         (unit_string, ex))

    # Check whether the result is a unit
    if not isinstance(evaluated_unit, Unit):
        if isinstance(evaluated_unit, Quantity):
            raise ValueError(('"%s" does not evaluate to a unit but to a '
                              'quantity -- make sure to only use units, e.g. '
                              '"siemens/metre**2" and not "1 * siemens/metre**2"') %
                             unit_string)
        else:
            raise ValueError(('"%s" does not evaluate to a unit, the result '
                             'has type %s instead.' % (unit_string,
                                                       type(evaluated_unit))))

    # No error has been raised, all good
    return evaluated_unit, FLOAT


def parse_string_equations(eqns):
    """
    Parse a string defining equations.
    
    Parameters
    ----------
    eqns : str
        The (possibly multi-line) string defining the equations. See the
        documentation of the `Equations` class for details.
    
    Returns
    -------
    equations : dict
        A dictionary mapping variable names to
        `~brian2.equations.equations.Equations` objects
    """
    equations = {}

    try:
        parsed = EQUATIONS.parseString(eqns, parseAll=True)
    except ParseException as p_exc:
        raise EquationError('Parsing failed: \n' + str(p_exc.line) + '\n' +
                            ' ' * (p_exc.column - 1) + '^\n' + str(p_exc))
    for eq in parsed:
        eq_type = eq.getName()
        eq_content = dict(eq.items())
        # Check for reserved keywords
        identifier = eq_content['identifier']

        # Convert unit string to Unit object
        unit, var_type = unit_and_type_from_string(eq_content['unit'])

        expression = eq_content.get('expression', None)
        if not expression is None:
            # Replace multiple whitespaces (arising from joining multiline
            # strings) with single space
            p = re.compile(r'\s{2,}')
            expression = Expression(p.sub(' ', expression))
        flags = list(eq_content.get('flags', []))

        equation = SingleEquation(eq_type, identifier, unit, var_type=var_type,
                                  expr=expression, flags=flags)

        if identifier in equations:
            raise EquationError('Duplicate definition of variable "%s"' %
                                identifier)

        equations[identifier] = equation

    return equations


class SingleEquation(object):
    '''
    Class for internal use, encapsulates a single equation or parameter.

    .. note::
        This class should never be used directly, it is only useful as part of
        the `Equations` class.
    
    Parameters
    ----------
    type : {PARAMETER, DIFFERENTIAL_EQUATION, SUBEXPRESSION}
        The type of the equation.
    varname : str
        The variable that is defined by this equation.
    unit : Unit
        The unit of the variable
    var_type : {FLOAT, INTEGER, BOOLEAN}
        The type of the variable (floating point value or boolean).
    expr : `Expression`, optional
        The expression defining the variable (or ``None`` for parameters).        
    flags: list of str, optional
        A list of flags that give additional information about this equation.
        What flags are possible depends on the type of the equation and the
        context.
    
    '''
    def __init__(self, type, varname, unit, var_type=FLOAT, expr=None,
                 flags=None):
        self.type = type
        self.varname = varname
        self.unit = unit
        self.var_type = var_type
        if not have_same_dimensions(unit, 1):
            if var_type == BOOLEAN:
                raise TypeError('Boolean variables are necessarily dimensionless.')
            elif var_type == INTEGER:
                raise TypeError('Integer variables are necessarily dimensionless.')

        if type == DIFFERENTIAL_EQUATION:
            if var_type != FLOAT:
                raise TypeError('Differential equations can only define floating point variables')
        self.expr = expr
        if flags is None:
            self.flags = []
        else:
            self.flags = flags

        # will be set later in the sort_subexpressions method of Equations
        self.update_order = -1


    identifiers = property(lambda self: self.expr.identifiers
                           if not self.expr is None else set([]),
                           doc='All identifiers in the RHS of this equation.')

    stochastic_variables = property(lambda self: set([variable for variable in self.identifiers
                                                      if variable =='xi' or variable.startswith('xi_')]),
                                    doc='Stochastic variables in the RHS of this equation')

    def _latex(self, *args):
        if self.type == DIFFERENTIAL_EQUATION:
            return (r'\frac{\mathrm{d}' + sympy.latex(self.varname) + r'}{\mathrm{d}t} = ' +
                    sympy.latex(str_to_sympy(self.expr.code)))
        elif self.type == SUBEXPRESSION:
            return (sympy.latex(self.varname) + ' = ' +
                    sympy.latex(str_to_sympy(self.expr.code)))
        elif self.type == PARAMETER:
            return sympy.latex(self.varname)

    def __str__(self):
        if self.type == DIFFERENTIAL_EQUATION:
            s = 'd' + self.varname + '/dt'
        else:
            s = self.varname

        if not self.expr is None:
            s += ' = ' + str(self.expr)

        s += ' : ' + str(self.unit)

        if len(self.flags):
            s += ' (' + ', '.join(self.flags) + ')'

        return s

    def __repr__(self):
        s = '<' + self.type + ' ' + self.varname

        if not self.expr is None:
            s += ': ' + self.expr.code

        s += ' (Unit: ' + str(self.unit)

        if len(self.flags):
            s += ', flags: ' + ', '.join(self.flags)

        s += ')>'
        return s

    def _repr_pretty_(self, p, cycle):
        '''
        Pretty printing for ipython.
        '''
        if cycle:
            # should never happen
            raise AssertionError('Cyclical call of SingleEquation._repr_pretty')

        if self.type == DIFFERENTIAL_EQUATION:
            p.text('d' + self.varname + '/dt')
        else:
            p.text(self.varname)

        if not self.expr is None:
            p.text(' = ')
            p.pretty(self.expr)

        p.text(' : ')
        p.pretty(self.unit)

        if len(self.flags):
            p.text(' (' + ', '.join(self.flags) + ')')

    def _repr_latex_(self):
        return '$' + sympy.latex(self) + '$'


class Equations(collections.Mapping):
    """
    Container that stores equations from which models can be created.
    
    String equations can be of any of the following forms:
    
    1. ``dx/dt = f : unit (flags)`` (differential equation)
    2. ``x = f : unit (flags)`` (equation)
    3. ``x : unit (flags)`` (parameter)

    String equations can span several lines and contain Python-style comments
    starting with ``#``    
    
    Parameters
    ----------
    eqs : `str` or list of `SingleEquation` objects
        A multiline string of equations (see above) -- for internal purposes
        also a list of `SingleEquation` objects can be given. This is done for
        example when adding new equations to implement the refractory
        mechanism. Note that in this case the variable names are not checked
        to allow for "internal names", starting with an underscore.
    kwds: keyword arguments
        Keyword arguments can be used to replace variables in the equation
        string. Arguments have to be of the form ``varname=replacement``, where
        `varname` has to correspond to a variable name in the given equation.
        The replacement can be either a string (replacing a name with a new
        name, e.g. ``tau='tau_e'``) or a value (replacing the variable name
        with the value, e.g. ``tau=tau_e`` or ``tau=10*ms``).
    """

    def __init__(self, eqns, **kwds):
        if isinstance(eqns, basestring):
            self._equations = parse_string_equations(eqns)
            # Do a basic check for the identifiers
            self.check_identifiers()
        else:
            self._equations = {}
            for eq in eqns:
                if not isinstance(eq, SingleEquation):
                    raise TypeError(('The list should only contain '
                                    'SingleEquation objects, not %s') % type(eq))
                if eq.varname in self._equations:
                    raise EquationError('Duplicate definition of variable "%s"' %
                                        eq.varname)
                self._equations[eq.varname] = eq

        # save these to change the keys of the dictionary later
        model_var_replacements = []
        for varname, replacement in kwds.iteritems():

            for eq in self.itervalues():
                # Replacing the name of a model variable (works only for strings)
                if eq.varname == varname:
                    if not isinstance(replacement, basestring):
                        raise ValueError(('Cannot replace model variable "%s" '
                                          'with a value') % varname)
                    if replacement in self:
                        raise EquationError(('Cannot replace model variable "%s" '
                                             'with "%s", duplicate definition '
                                             'of "%s".' % (varname, replacement,
                                                           replacement)))
                    # make sure that the replacement is a valid identifier
                    Equations.check_identifier(replacement)
                    eq.varname = replacement
                    model_var_replacements.append((varname, replacement))

                if varname in eq.identifiers:
                    if isinstance(replacement, basestring):
                        # replace the name with another name
                        new_code = re.sub('\\b' + varname + '\\b',
                                          replacement, eq.expr.code)
                    else:
                        # replace the name with a value
                        new_code = re.sub('\\b' + varname + '\\b',
                                          '(' + repr(replacement) + ')',
                                          eq.expr.code)
                    try:
                        eq.expr = Expression(new_code)
                    except ValueError as ex:
                        raise ValueError(('Replacing "%s" with "%r" failed: %s') %
                                         (varname, replacement, ex))

        # For change in model variable names, we have already changed the
        # varname attribute of the SingleEquation object, but not the key of
        # our dicitionary
        for varname, replacement in model_var_replacements:
            self._equations[replacement] = self._equations.pop(varname)

        # Check for special symbol xi (stochastic term)
        uses_xi = None
        for eq in self._equations.itervalues():
            if not eq.expr is None and 'xi' in eq.expr.identifiers:
                if not eq.type == DIFFERENTIAL_EQUATION:
                    raise EquationError(('The equation defining %s contains the '
                                         'symbol "xi" but is not a differential '
                                         'equation.') % eq.varname)
                elif not uses_xi is None:
                    raise EquationError(('The equation defining %s contains the '
                                         'symbol "xi", but it is already used '
                                         'in the equation defining %s.') %
                                        (eq.varname, uses_xi))
                else:
                    uses_xi = eq.varname

        # rearrange subexpressions
        self._sort_subexpressions()

    def __iter__(self):
        return iter(self._equations)

    def __len__(self):
        return len(self._equations)

    def __getitem__(self, key):
        return self._equations[key]

    def __add__(self, other_eqns):
        if isinstance(other_eqns, basestring):
            other_eqns = parse_string_equations(other_eqns)
        elif not isinstance(other_eqns, Equations):
            return NotImplemented

        return Equations(self.values() + other_eqns.values())

    #: A set of functions that are used to check identifiers (class attribute).
    #: Functions can be registered with the static method
    #: `Equations.register_identifier_check` and will be automatically
    #: used when checking identifiers
    identifier_checks = {check_identifier_basic, check_identifier_reserved,
                         check_identifier_functions, check_identifier_units}

    @staticmethod
    def register_identifier_check(func):
        '''
        Register a function for checking identifiers.
        
        Parameters
        ----------
        func : callable
            The function has to receive a single argument, the name of the
            identifier to check, and raise a ValueError if the identifier
            violates any rule.

        '''
        if not hasattr(func, '__call__'):
            raise ValueError('Can only register callables.')

        Equations.identifier_checks.add(func)

    @staticmethod
    def check_identifier(identifier):
        '''
        Perform all the registered checks. Checks can be registered via
        `Equations.register_identifier_check`.
        
        Parameters
        ----------
        identifier : str
            The identifier that should be checked
        
        Raises
        ------
        ValueError
            If any of the registered checks fails.
        '''
        for check_func in Equations.identifier_checks:
            check_func(identifier)

    def check_identifiers(self):
        '''
        Check all identifiers for conformity with the rules.
        
        Raises
        ------
        ValueError
            If an identifier does not conform to the rules.
        
        See also
        --------
        Equations.check_identifier : The function that is called for each identifier.
        '''
        for name in self.names:
            Equations.check_identifier(name)

    def get_substituted_expressions(self, variables=None):
        '''
        Return a list of ``(varname, expr)`` tuples, containing all
        differential equations with all the subexpression variables
        substituted with the respective expressions.

        Parameters
        ----------
        variables : dict, optional
            A mapping of variable names to `Variable`/`Function` objects.

        Returns
        -------
        expr_tuples : list of (str, `CodeString`)
            A list of ``(varname, expr)`` tuples, where ``expr`` is a
            `CodeString` object with all subexpression variables substituted
            with the respective expression.
        '''

        subst_exprs = []
        substitutions = {}
        for eq in self.ordered:
            # Skip parameters
            if eq.expr is None:
                continue

            new_sympy_expr = str_to_sympy(eq.expr.code, variables).xreplace(substitutions)
            new_str_expr = sympy_to_str(new_sympy_expr)
            expr = Expression(new_str_expr)

            if eq.type == SUBEXPRESSION:
                substitutions.update({sympy.Symbol(eq.varname, real=True): str_to_sympy(expr.code, variables)})
            elif eq.type == DIFFERENTIAL_EQUATION:
                #  a differential equation that we have to check
                subst_exprs.append((eq.varname, expr))
            else:
                raise AssertionError('Unknown equation type %s' % eq.type)

        return subst_exprs

    def _get_stochastic_type(self):
        '''
        Returns the type of stochastic differential equations (additivive or
        multiplicative). The system is only classified as ``additive`` if *all*
        equations have only additive noise (or no noise).
        
        Returns
        -------
        type : str
            Either ``None`` (no noise variables), ``'additive'`` (factors for
            all noise variables are independent of other state variables or
            time), ``'multiplicative'`` (at least one of the noise factors
            depends on other state variables and/or time).
        '''
        
        # TODO: Add debug output
        
        if not self.is_stochastic:
            return None
        
        for _, expr in self.get_substituted_expressions():
            _, stochastic = expr.split_stochastic()
            if stochastic is not None:
                for factor in stochastic.itervalues():
                    if 't' in factor.identifiers:
                        # noise factor depends on time
                        return 'multiplicative'

                    for identifier in factor.identifiers:
                        if identifier in self.diff_eq_names:
                            # factor depends on another state variable
                            return 'multiplicative'
        
        return 'additive'


    ############################################################################
    # Properties
    ############################################################################

    # Lists of equations or (variable, expression tuples)
    ordered = property(lambda self: sorted(self._equations.itervalues(),
                                           key=lambda key: key.update_order),
                                           doc='A list of all equations, sorted '
                                           'according to the order in which they should '
                                           'be updated')

    diff_eq_expressions = property(lambda self: [(varname, eq.expr) for
                                                 varname, eq in self.iteritems()
                                                 if eq.type == DIFFERENTIAL_EQUATION],
                                  doc='A list of (variable name, expression) '
                                  'tuples of all differential equations.')

    eq_expressions = property(lambda self: [(varname, eq.expr) for
                                            varname, eq in self.iteritems()
                                            if eq.type in (SUBEXPRESSION,
                                                              DIFFERENTIAL_EQUATION)],
                              doc='A list of (variable name, expression) '
                                  'tuples of all equations.')

    # Sets of names

    names = property(lambda self: set([eq.varname for eq in self.ordered]),
                     doc='All variable names defined in the equations.')

    diff_eq_names = property(lambda self: set([eq.varname for eq in self.ordered
                                           if eq.type == DIFFERENTIAL_EQUATION]),
                             doc='All differential equation names.')

    subexpr_names = property(lambda self: set([eq.varname for eq in self.ordered
                                               if eq.type == SUBEXPRESSION]),
                             doc='All subexpression names.')

    eq_names = property(lambda self: set([eq.varname for eq in self.ordered
                                           if eq.type in (DIFFERENTIAL_EQUATION,
                                                          SUBEXPRESSION)]),
                        doc='All equation names (including subexpressions).')

    parameter_names = property(lambda self: set([eq.varname for eq in self.ordered
                                             if eq.type == PARAMETER]),
                               doc='All parameter names.')

    units = property(lambda self:dict([(var, eq.unit) for var, eq in
                                       self._equations.iteritems()]),
                     doc='Dictionary of all internal variables and their corresponding units.')


    identifiers = property(lambda self: set().union(*[eq.identifiers for
                                                      eq in self._equations.itervalues()]) -
                           self.names,
                           doc=('Set of all identifiers used in the equations, '
                                'excluding the variables defined in the equations'))

    stochastic_variables = property(lambda self: set([variable for variable in self.identifiers
                                                      if variable =='xi' or variable.startswith('xi_')]))

    # general properties
    is_stochastic = property(lambda self: len(self.stochastic_variables) > 0,
                             doc='Whether the equations are stochastic.')

    stochastic_type = property(fget=_get_stochastic_type)

    def _sort_subexpressions(self):
        '''
        Sorts the subexpressions in a way that resolves their dependencies
        upon each other. After this method has been run, the subexpressions
        returned by the ``ordered`` property are in the order in which
        they should be updated
        '''

        # Get a dictionary of all the dependencies on other subexpressions,
        # i.e. ignore dependencies on parameters and differential equations
        static_deps = {}
        for eq in self._equations.itervalues():
            if eq.type == SUBEXPRESSION:
                static_deps[eq.varname] = [dep for dep in eq.identifiers if
                                           dep in self._equations and
                                           self._equations[dep].type == SUBEXPRESSION]
        
        try:
            sorted_eqs = topsort(static_deps)
        except ValueError:
            raise ValueError('Cannot resolve dependencies between static '
                             'equations, dependencies contain a cycle.')

        # put the equations objects in the correct order
        for order, static_variable in enumerate(sorted_eqs):
            self._equations[static_variable].update_order = order

        # Sort differential equations and parameters after subexpressions
        for eq in self._equations.itervalues():
            if eq.type == DIFFERENTIAL_EQUATION:
                eq.update_order = len(sorted_eqs)
            elif eq.type == PARAMETER:
                eq.update_order = len(sorted_eqs) + 1

    def check_units(self, group, run_namespace=None, level=0):
        '''
        Check all the units for consistency.
        
        Parameters
        ----------
        group : `Group`
            The group providing the context
        run_namespace : dict, optional
            A namespace provided to the `Network.run` function.
        level : int, optional
            How much further to go up in the stack to find the calling frame

        Raises
        ------
        DimensionMismatchError
            In case of any inconsistencies.
        '''
        all_variables = dict(group.variables)
        external = frozenset().union(*[expr.identifiers
                                     for _, expr in self.eq_expressions])
        external -= set(all_variables.keys())

        resolved_namespace = group.resolve_all(external,
                                               external,  # all variables are user defined
                                               run_namespace=run_namespace,
                                               level=level+1)
        all_variables.update(resolved_namespace)
        for var, eq in self._equations.iteritems():
            if eq.type == PARAMETER:
                # no need to check units for parameters
                continue

            if eq.type == DIFFERENTIAL_EQUATION:
                try:
                    check_unit(str(eq.expr), self.units[var] / second,
                               all_variables)
                except DimensionMismatchError as ex:
                    raise DimensionMismatchError(('Inconsistent units in '
                                                  'differential equation '
                                                  'defining variable %s:'
                                                  '\n%s') % (eq.varname,
                                                             ex.desc),
                                                 *ex.dims)
            elif eq.type == SUBEXPRESSION:
                try:
                    check_unit(str(eq.expr), self.units[var],
                           all_variables)
                except DimensionMismatchError as ex:
                    raise DimensionMismatchError(('Inconsistent units in '
                                                  'subexpression %s:'
                                                  '\n%s') % (eq.varname,
                                                             ex.desc),
                                                 *ex.dims)
            else:
                raise AssertionError('Unknown equation type: "%s"' % eq.type)


    def check_flags(self, allowed_flags):
        '''
        Check the list of flags.
        
        Parameters
        ----------
        allowed_flags : dict
             A dictionary mapping equation types (PARAMETER,
             DIFFERENTIAL_EQUATION, SUBEXPRESSION) to a list of strings (the
             allowed flags for that equation type)
        
        Notes
        -----
        Not specifying allowed flags for an equation type is the same as
        specifying an empty list for it.
        
        Raises
        ------
        ValueError
            If any flags are used that are not allowed.
        '''
        for eq in self.itervalues():
            for flag in eq.flags:
                if not eq.type in allowed_flags or len(allowed_flags[eq.type]) == 0:
                    raise ValueError('Equations of type "%s" cannot have any flags.' % eq.type)
                if not flag in allowed_flags[eq.type]:
                    raise ValueError(('Equations of type "%s" cannot have a '
                                      'flag "%s", only the following flags '
                                      'are allowed: %s') % (eq.type,
                                                            flag, allowed_flags[eq.type]))

    ############################################################################
    # Representation
    ############################################################################

    def __str__(self):
        strings = [str(eq) for eq in self.ordered]
        return '\n'.join(strings)

    def __repr__(self):
        return '<Equations object consisting of %d equations>' % len(self._equations)

    def _latex(self, *args):        
        equations = []
        t = sympy.Symbol('t')
        for eq in self._equations.itervalues():
            # do not use SingleEquations._latex here as we want nice alignment
            varname = sympy.Symbol(eq.varname)
            if eq.type == DIFFERENTIAL_EQUATION:
                lhs = r'\frac{\mathrm{d}' + sympy.latex(varname) + r'}{\mathrm{d}t}'
            else:
                # Normal equation or parameter
                lhs = varname
            if not eq.type == PARAMETER:
                rhs = str_to_sympy(eq.expr.code)
            if len(eq.flags):
                flag_str = ', flags: ' + ', '.join(eq.flags)
            else:
                flag_str = ''
            if eq.type == PARAMETER:
                eq_latex = r'%s &&& \text{(unit: $%s$%s)}' % (sympy.latex(lhs),                                 
                                                              sympy.latex(eq.unit),
                                                              flag_str)
            else:
                eq_latex = r'%s &= %s && \text{(unit of $%s$: $%s$%s)}' % (sympy.latex(lhs),
                                                                           sympy.latex(rhs),
                                                                           sympy.latex(varname),
                                                                           sympy.latex(eq.unit),
                                                                           flag_str)
            equations.append(eq_latex)
        return r'\begin{align*}' + (r'\\' + '\n').join(equations) + r'\end{align*}'

    def _repr_latex_(self):
        return sympy.latex(self)

    def _repr_pretty_(self, p, cycle):
        ''' Pretty printing for ipython '''
        if cycle:
            # Should never happen
            raise AssertionError('Cyclical call of Equations._repr_pretty_')
        for eq in self._equations.itervalues():
            p.pretty(eq)
            p.breakable('\n')
