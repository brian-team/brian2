'''
Differential equations for Brian models.
'''
import collections
import keyword
import re
import string

from pyparsing import (Group, ZeroOrMore, OneOrMore, Optional, Word, CharsNotIn,
                       Combine, Suppress, restOfLine, LineEnd, ParseException)

from brian2.units.fundamentalunits import DimensionMismatchError
from brian2.units.allunits import second
from brian2.utils.stringtools import word_substitute
from brian2.utils.logger import get_logger

from .codestrings import Expression
from .unitcheck import get_unit_from_string

__all__ = ['Equations']

logger = get_logger(__name__)

# Equation types (currently simple strings but always use the constants,
# this might get refactored into objects, for example)
PARAMETER = 'parameter'
DIFFERENTIAL_EQUATION = 'differential equation'
STATIC_EQUATION = 'static equation'


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
UNIT = Word(string.ascii_letters + string.digits + '*/. ').setResultsName('unit')

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
PARAMETER_EQ = Group(IDENTIFIER + Suppress(':') + UNIT +
                     Optional(FLAGS)).setResultsName(PARAMETER)

# Static equation:
# x = 2 * y : volt (flags)
STATIC_EQ = Group(IDENTIFIER + Suppress('=') + EXPRESSION + Suppress(':') +
                  UNIT + Optional(FLAGS)).setResultsName(STATIC_EQUATION)

# Differential equation
# dx/dt = -x / tau : volt
DIFF_OP = (Suppress('d') + IDENTIFIER + Suppress('/') + Suppress('dt'))
DIFF_EQ = Group(DIFF_OP + Suppress('=') + EXPRESSION + Suppress(':') + UNIT +
                Optional(FLAGS)).setResultsName(DIFFERENTIAL_EQUATION)

# ignore comments
EQUATION = (PARAMETER_EQ | STATIC_EQ | DIFF_EQ).ignore('#' + restOfLine)
EQUATIONS = ZeroOrMore(EQUATION)

class EquationError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

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
    special variables are: 't', 'dt', and 'xi'.
    
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


def check_identifier(identifier):
    '''
    Perform all the registered checks. Checks can be registered via
    :func:`Equations.register_identifier_check`.
    
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
                            ' '*(p_exc.column - 1) + '^\n' + str(p_exc))
    for eq in parsed:
        eq_type = eq.getName()
        eq_content = dict(eq.items())
        # Check for reserved keywords
        identifier = eq_content['identifier']
        
        # Convert unit string to Unit object
        unit = get_unit_from_string(eq_content['unit'])
        
        expression = eq_content.get('expression', None)
        if not expression is None:
            # Replace multiple whitespaces (arising from joining multiline
            # strings) with single space
            p = re.compile(r'\s{2,}')
            expression = Expression(p.sub(' ', expression))
        flags = list(eq_content.get('flags', []))

        equation = SingleEquation(eq_type, identifier, unit, expression, flags) 
        
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
    eq_type : {PARAMETER, DIFFERENTIAL_EQUATION, STATIC_EQUATION}
        The type of the equation.
    varname : str
        The variable that is defined by this equation.
    unit : Unit
        The unit of the variable
    expr : `Expression`, optional
        The expression defining the variable (or ``None`` for parameters).        
    flags: list of str, optional
        A list of flags that give additional information about this equation.
        What flags are possible depends on the type of the equation and the
        context.
    
    '''
    def __init__(self, eq_type, varname, unit, expr=None, flags=None):
        self.eq_type = eq_type
        self.varname = varname
        self.unit = unit        
        self.expr = expr
        if flags is None:
            self.flags = []
        else:
            self.flags = flags
        
        # will be set later in the sort_static_equations method of Equations
        self.update_order = -1

    def replace_code(self, code):
        '''
        Return a new `SingleEquation` based on an existing one. This is used
        internally, when an equation string is replaced or changed while all
        the other information is kept (units, flags, etc.). For example,
        the `~brian2.equations.refractory.add_refractory` function replaces
        all differential equations having the ``(active)`` flag with a new
        equation. 
        '''
        return SingleEquation(self.eq_type,
                              self.varname,
                              self.unit,
                              self.expr.replace_code(code),                              
                              self.flags)

    identifiers = property(lambda self: self.expr.identifiers
                           if not self.expr is None else set([]),
                           doc='All identifiers in the RHS of this equation.')
        

    def __str__(self):
        if self.eq_type == DIFFERENTIAL_EQUATION:
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
        s = '<' + self.eq_type + ' ' + self.varname
        
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
        
        if self.eq_type == DIFFERENTIAL_EQUATION:
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
                    check_identifier(replacement)
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
                if not eq.eq_type == DIFFERENTIAL_EQUATION:
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
        
        # rearrange static equations
        self._sort_static_equations()

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
    #: `:meth:Equations.register_identifier_check` and will be automatically
    #: used when checking identifiers
    identifier_checks = set([check_identifier_basic,
                             check_identifier_reserved])
    
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

    def check_identifiers(self):
        '''
        Check all identifiers for conformity with the rules.
        
        Raises
        ------
        ValueError
            If an identifier does not conform to the rules.
        
        See also
        --------
        brian2.equations.equations.check_identifier : The function that is called for each identifier.
        '''
        for name in self.names:
            check_identifier(name)

    def _get_substituted_expressions(self):
        '''
        Return a list of ``(varname, expr)`` tuples, containing all
        differential equations with all the static equation variables
        substituted with the respective expressions.
        
        Returns
        -------
        expr_tuples : list of (str, `CodeString`)
            A list of ``(varname, expr)`` tuples, where ``expr`` is a
            `CodeString` object with all static equation variables substituted
            with the respective expression.
        '''
        subst_exprs = []
        substitutions = {}        
        for eq in self.ordered:
            # Skip parameters
            if eq.expr is None:
                continue
            
            expr = eq.expr.replace_code(word_substitute(eq.expr.code, substitutions))
            
            if eq.eq_type == STATIC_EQUATION:
                substitutions.update({eq.varname: '(%s)' % expr.code})
            elif eq.eq_type == DIFFERENTIAL_EQUATION:
                #  a differential equation that we have to check                                
                expr.resolve(self.names)
                subst_exprs.append((eq.varname, expr))
            else:
                raise AssertionError('Unknown equation type %s' % eq.eq_type)
        
        return subst_exprs        

    def _get_units(self):
        '''
        Dictionary of all internal variables and their corresponding units.
        '''
        return dict([(var, eq.unit) for var, eq in
                      self._equations.iteritems()])

    # Properties
    
    ordered = property(lambda self: sorted(self._equations.itervalues(),
                                           key=lambda key: key.update_order),
                                           doc='A list of all equations, sorted '
                                           'according to the order in which they should '
                                           'be updated')
    
    diff_eq_expressions = property(lambda self: [(varname, eq.expr) for 
                                                 varname, eq in self.iteritems()
                                                 if eq.eq_type == DIFFERENTIAL_EQUATION],
                                  doc='A list of (variable name, expression) '
                                  'tuples of all differential equations.')
    
    eq_expressions = property(lambda self: [(varname, eq.expr) for 
                                            varname, eq in self.iteritems()
                                            if eq.eq_type in (STATIC_EQUATION,
                                                              DIFFERENTIAL_EQUATION)],
                                  doc='A list of (variable name, expression) '
                                  'tuples of all equations.') 
    
    substituted_expressions = property(_get_substituted_expressions)
    
    names = property(lambda self: set([eq.varname for eq in self.ordered]),
                     doc='All variable names defined in the equations.')
    
    diff_eq_names = property(lambda self: [eq.varname for eq in self.ordered
                                           if eq.eq_type == DIFFERENTIAL_EQUATION],
                             doc='All differential equation names.')
    static_eq_names = property(lambda self: [eq.varname for eq in self.ordered
                                           if eq.eq_type == STATIC_EQUATION],
                               doc='All static equation names.')
    eq_names = property(lambda self: [eq.varname for eq in self.ordered
                                           if eq.eq_type in (DIFFERENTIAL_EQUATION, STATIC_EQUATION)],
                        doc='All (static and differential) equation names.')
    parameter_names = property(lambda self: [eq.varname for eq in self.ordered
                                             if eq.eq_type == PARAMETER],
                               doc='All parameter names.')    
        
    units = property(_get_units)
    
    variables = property(lambda self: set(self.units.keys()),
                         doc='Set of all variables (including t, dt, and xi)')
    
    identifiers = property(lambda self: set().union(*[eq.identifiers for
                                                      eq in self._equations.itervalues()]) -
                           self.names,
                           doc=('Set of all identifiers used in the equations, '
                                'excluding the variables defined in the equations'))
    
    def _sort_static_equations(self):
        '''
        Sorts the static equations in a way that resolves their dependencies
        upon each other. After this method has been run, the static equations
        returned by the ``ordered`` property are in the order in which
        they should be updated
        '''
        
        # Get a dictionary of all the dependencies on other static equations,
        # i.e. ignore dependencies on parameters and differential equations
        static_deps = {}
        for eq in self._equations.itervalues():
            if eq.eq_type == STATIC_EQUATION:
                static_deps[eq.varname] = [dep for dep in eq.identifiers if
                                           dep in self._equations and
                                           self._equations[dep].eq_type == STATIC_EQUATION]
        
        # Use the standard algorithm for topological sorting:
        # http://en.wikipedia.org/wiki/Topological_sorting
                
        # List that will contain the sorted elements
        sorted_eqs = [] 
        # set of all nodes with no incoming edges:
        no_incoming = set([var for var, deps in static_deps.iteritems()
                           if len(deps) == 0]) 
        
        while len(no_incoming):
            n = no_incoming.pop()
            sorted_eqs.append(n)
            # find variables m depending on n
            dependent = [m for m, deps in static_deps.iteritems()
                         if n in deps]
            for m in dependent:
                static_deps[m].remove(n)
                if len(static_deps[m]) == 0:
                    # no other dependencies
                    no_incoming.add(m)
        if any([len(deps) > 0 for deps in static_deps.itervalues()]):
            raise ValueError('Cannot resolve dependencies between static '
                             'equations, dependencies contain a cycle.')
        
        # put the equations objects in the correct order
        for order, static_variable in enumerate(sorted_eqs):
            self._equations[static_variable].update_order = order
        
        # Sort differential equations and parameters after static equations
        for eq in self._equations.itervalues():
            if eq.eq_type == DIFFERENTIAL_EQUATION:
                eq.update_order = len(sorted_eqs)
            elif eq.eq_type == PARAMETER:
                eq.update_order = len(sorted_eqs) + 1

    def check_units(self, namespace, specifiers):
        '''
        Check all the units for consistency.
        
        Parameters
        ----------
        namespace : `CompoundNamespace`
            The namespace for resolving external identifiers, should be
            provided by the `NeuronGroup` or `Synapses`.
        specifiers: dict of `Specifier` objects
            The specifiers of the state variables and internal variables
            (e.g. t and dt)
        
        Raises
        ------
        DimensionMismatchError
            In case of any inconsistencies.
        '''
        
        # Create a mapping with all identifier names to either their actual
        # value (for external identifiers) or their unit (for specifiers)
        unit_namespace = {}
        for name in self.identifiers | self.variables:
            if name in specifiers:
                unit_namespace.update({name: specifiers[name].unit})
            else:
                # This raises an error if the identifier cannot be resolved
                unit_namespace.update({name: namespace[name]})            
                
        for var, eq in self._equations.iteritems():
            if eq.eq_type == PARAMETER:
                # no need to check units for parameters
                continue
            
            if eq.eq_type == DIFFERENTIAL_EQUATION:
                try:
                    eq.expr.check_units(self.units[var] / second, unit_namespace)
                except DimensionMismatchError as dme:
                    raise DimensionMismatchError(('Differential equation defining '
                                                  '%s does not use consistent units: %s') % 
                                                 (var, dme.desc), *dme.dims)
            elif eq.eq_type == STATIC_EQUATION:
                try:
                    eq.expr.check_units(self.units[var], unit_namespace)
                except DimensionMismatchError as dme:
                    raise DimensionMismatchError(('Static equation defining '
                                                  '%s does not use consistent units: %s') % 
                                                 (var, dme.desc), *dme.dims)                
            else:
                raise AssertionError('Unknown equation type: "%s"' % eq.eq_type)


    def check_flags(self, allowed_flags):
        '''
        Check the list of flags.
        
        Parameters
        ----------
        allowed_flags : dict
             A dictionary mapping equation types (PARAMETER,
             DIFFERENTIAL_EQUATION, STATIC_EQUATION) to a list of strings (the
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
                if not eq.eq_type in allowed_flags or len(allowed_flags[eq.eq_type]) == 0:
                    raise ValueError('Equations of type "%s" cannot have any flags.' % eq.eq_type)
                if not flag in allowed_flags[eq.eq_type]:
                    raise ValueError(('Equations of type "%s" cannot have a '
                                      'flag "%s", only the following flags '
                                      'are allowed: %s') % (eq.eq_type,
                                                            flag, allowed_flags[eq.eq_type]))

    #
    # Representation
    # 

    def __str__(self):
        strings = [str(eq) for eq in self.ordered]
        return '\n'.join(strings)
    
    def __repr__(self):
        return '<Equations object consisting of %d equations>' % len(self._equations)

    def _repr_pretty_(self, p, cycle):
        ''' Pretty printing for ipython '''
        if cycle: 
            # Should never happen
            raise AssertionError('Cyclical call of Equations._repr_pretty_')
        for eq in self._equations.itervalues():
            p.pretty(eq)
            p.breakable('\n')
