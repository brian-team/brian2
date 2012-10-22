'''
Module defining `CodeString`, a class for a string of code together with
information about its namespace. Only serves as a parent class, its subclasses
`Expression` and `Statements` are the ones that are actually used.
'''

import inspect

import sympy
from sympy.core.sympify import SympifyError

from .unitcheck import get_default_unit_namespace, SPECIAL_VARS

from brian2 import get_dimensions, DimensionMismatchError, get_logger
from brian2.utils.stringtools import get_identifiers, word_substitute

__all__ = ['Expression', 'Statements']

logger = get_logger(__name__)


def _conflict_warning(message, resolutions, logger):
    '''
    A little helper functions to generate warnings for logging. Specific
    to the `CodeString.resolve` method and should only be used by it.
    
    Parameters
    ----------
    message : str
        The first part of the warning message.
    resolutions : list of str
        A list of (namespace, object) tuples.
    logger : `BrianLogger`
        The logger object.
    '''
    if len(resolutions) == 0:
        # nothing to warn about
        return
    elif len(resolutions) == 1:
        second_part = ('but also refers to a variable in the %s namespace:'
                       ' %r') % (resolutions[0][0], resolutions[0][1])
    elif len(resolutions) > 1:
        second_part = ('but also refers to a variable in the following '
                       'namespaces: %s') % (', '.join([r[0] for r in resolutions]))
    
    logger.warn(message + ' ' + second_part,
                'CodeString.resolve.resolution_conflict')


class CodeString(object):
    '''
    A class for representing strings and an attached namespace.
    
    Parameters
    ----------
    code : str
        The code string, may be an expression or a statement(s) (possibly
        multi-line).
    namespace : dict, optional
        A dictionary mapping identifiers (strings) to objects. Will be used as
        a namespace for the `code`.
    exhaustive : bool, optional
        If set to ``True`` (the default), no local/global namespace will be
        saved, meaning that the given namespace has to be exhaustive (except
        for units). If set to ``False``, the given `namespace` augments the
        local and global namespace (taking precedence over them in case of
        conflicting definitions).
    level : int, optional
        The level in the stack (an integer >=0) where to look for locals
        and globals    
    
    Notes
    -----
    If `exhaustive` is not ``False`` (meaning that the namespace for the string
    is explicitly specified), the `CodeString` object saves a copy of the
    current local and global namespace for later use in resolving identifiers.
    '''

    def __init__(self, code, namespace=None, exhaustive=True, level=0):
        
        self._code = code
        
        # extract identifiers from the code
        self._identifiers = set(get_identifiers(code))
        
        self._exhaustive = exhaustive
        
        if namespace is None or not exhaustive:
            frame = inspect.stack()[level + 1][0]
            self._locals = frame.f_locals.copy()
            self._globals = frame.f_globals.copy()
        else:
            self._locals = {}
            self._globals = {}
        
        self._given_namespace = namespace
        
        # The namespace containing resolved references
        self._namespace = None
    
    code = property(lambda self: self._code,
                    doc='The code string.')

    exhaustive = property(lambda self: self._exhaustive and not self._given_namespace is None,
                          doc='Whether the namespace is exhaustively defined.')
        
    identifiers = property(lambda self: self._identifiers,
                           doc='Set of identifiers in the code string.')
    
    is_resolved = property(lambda self: not self._namespace is None,
                           doc='Whether the external identifiers have been resolved.')
        
    namespace = property(lambda self: self._namespace,
                         doc='The namespace resolving external identifiers.')


    def resolve(self, internal_variables):
        '''
        Determine the namespace, containing resolved references to externally
        defined variables and functions.

        Parameters
        ----------
        internal_variables : list of str
            A list of variables that should not be resolved.
        
        Notes
        -----
        The resulting namespace includes units but does not include anything
        present in the `internal variables` collection.
        
        Raises
        ------
        ValueError
            If a variable/function cannot be resolved and is not contained in
            `internal_variables`.
        '''

        if self.is_resolved:
            raise TypeError('Variables have already been resolved before.')

        unit_namespace = get_default_unit_namespace()
        
        namespace = {}
        for identifier in self.identifiers:
            # We save tuples of (namespace description, referred object) to
            # give meaningful warnings in case of duplicate definitions
            matches = []
            if (not self._given_namespace is None and
                identifier in self._given_namespace):
                matches.append(('user-defined',
                                self._given_namespace[identifier]))
            if identifier in self._locals:
                matches.append(('locals',
                                self._locals[identifier]))
            if identifier in self._globals:
                matches.append(('globals',
                                self._globals[identifier]))
            if identifier in unit_namespace:
                matches.append(('units',
                               unit_namespace[identifier]))
            
            # raise warnings in case of conflicts
            if identifier in SPECIAL_VARS:
                # The identifier is t, dt, or xi
                _conflict_warning(('The name "%s" in the code string "%s" '
                                  'has a special meaning') %
                                  (identifier, self.code), matches, logger)
            elif identifier in internal_variables:
                # The identifier is an internal variable
                _conflict_warning(('The name "%s" in the code string "%s" '
                                   'refers to an internal variable') %
                                  (identifier, self.code),
                                  matches, logger)
            else:
                # The identifier is not an internal variable
                if len(matches) == 0:
                    raise ValueError(('The identifier "%s" in the code string '
                                     '"%s" could not be resolved.') % 
                                     (identifier, self.code))
                elif len(matches) > 1:
                    # Possibly, all matches refer to the same object
                    first_obj = matches[0][1]
                    if not all([m[1] is first_obj for m in matches]):
                        _conflict_warning(('The name "%s" in the code string "%s" '
                                     'refers to different objects in different '
                                     'namespaces used for resolving. Will use '
                                     'the object from the %s namespace (%r)') %
                                    (identifier, self.code, matches[0][0],
                                     first_obj), matches[1:], logger)
                
                # use the first match (according to resolution order)
                namespace[identifier] = matches[0][1]
                
        self._namespace = namespace

    def frozen(self):
        '''
        Replace all external variables by their floating point values.
        
        Returns
        -------
        frozen : `CodeString`
            A new `CodeString` object, where all external variables are replaced
            by their floating point values and removed from the namespace.
        
        Notes
        -----
        The namespace has to be resolved using the
        `~brian2.equations.codestrings.CodeString.resolve` method first.
        '''
        
        if not self.is_resolved:
            raise TypeError('Can only freeze resolved CodeString objects.')
        
        #TODO: For expressions, this could be done more elegantly with sympy
        
        new_namespace = self.namespace.copy()
        substitutions = {}
        for identifier in self.identifiers:
            if identifier in new_namespace:
                # Try to replace the variable with its float value
                try:
                    float_value = float(new_namespace[identifier])
                    substitutions[identifier] = str(float_value)
                    # Reference in namespace no longer needed
                    del new_namespace[identifier]
                except (ValueError, TypeError):
                    pass
        
        # Apply the substitutions to the string
        new_code = word_substitute(self.code, substitutions)
        
        # Create a new CodeString object with the new code and namespace
        new_obj = type(self)(new_code, namespace=new_namespace,
                             exhaustive=True)
        new_obj._namespace = new_namespace.copy()
        return new_obj

    def __str__(self):
        return self.code
    
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.code)


class Statements(CodeString):
    '''
    Class for representing statements and their associated namespace.

    Parameters
    ----------
    code : str
        The statement or statements. Several statements can be given as a
        multi-line string or separated by semicolons.
    namespace : dict, optional
        A dictionary mapping identifiers (strings) to objects
        (see `~brian2.equations.codestrings.CodeString` for more details).
    exhaustive : bool, optional
        Whether the given namespace is exhaustive
        (see `~brian2.equations.codestrings.CodeString` for more details).        
    level : int, optional
        The level in the stack (an integer >=0) where to look for locals
        and globals (see `~brian2.equations.codestrings.CodeString` for more
        details).
    
    Notes
    -----
    Currently, the implementation of this class does not add anything to
    `~brian2.equations.codestrings.CodeString`, but it should be used instead
    of that class for clarity and to allow for future functionality that is
    only relevant to statements and not to expressions.
    '''
    pass

class Expression(CodeString):
    '''
    Class for representing an expression and its associated namespace.

    Parameters
    ----------
    code : str
        The expression. Note that the expression has to be written in a form
        that is parseable by sympy.
    namespace : dict, optional
        A dictionary mapping identifiers (strings) to objects
        (see `~brian2.equations.codestrings.CodeString` for more details).
    exhaustive : bool, optional
        Whether the given namespace is exhaustive
        (see `~brian2.equations.codestrings.CodeString` for more details).        
    level : int, optional
        The level in the stack (an integer >=0) where to look for locals
        and globals (see `~brian2.equations.codestrings.CodeString` for more
        details).
    '''
    
    def __init__(self, code, namespace=None, exhaustive=True, level=0):
        super(Expression, self).__init__(code, namespace, exhaustive, level + 1)
        
        try:
            self._sympy_expr = sympy.sympify(self.code)
        except SympifyError:
            raise SyntaxError('Expression "%s" cannot be parsed with sympy' %
                              self.code)
         
    
    def check_linearity(self, variable):
        '''
        Return whether the expression is linear.
        
        Parameters
        ----------
        variable : str
            The variable name against which linearity is checked.
        
        Returns
        -------
        linear : bool
            Whether the expression  is linear with respect to `variable`,
            assuming that all other variables are constants.
        
        '''

    
        x = sympy.Symbol(variable)
    
        if not x in self._sympy_expr:
            return True
    
    #    # This tries to check whether the expression can be rewritten in an a*x + b
    #    # but apparently this does not work very well
    #    a = Wild('a', exclude=[x])
    #    b = Wild('b', exclude=[x])
    #    matches = sympy_expr.match(a * x + b) 
    #
    #    return not matches is None
    
        # This seems to be more robust: Take the derivative with respect to the
        # variable
        diff_f = sympy.diff(self._sympy_expr, x).simplify()
    
        # if the expression is linear, x should have disappeared
        return not x in diff_f
    
    def eval(self, internal_variables):
        '''
        Evaluate the expression in its namespace. The namespace is augmented by
        the values given in `internal_variables`.
        
        Parameters
        ----------
        internal_variables : dict
            A dictionary mapping variable names to their values.
        
        Returns
        -------
        result
            The result of evaluating the expression.

        Notes
        -----
        The namespace has to be resolved using the
        `~brian2.equations.codestrings.CodeString.resolve` method first.        
        '''
    
        if not self.is_resolved:
            raise TypeError('Can only evaluate resolved CodeString objects.')
        
        namespace = self.namespace.copy()
        namespace.update(internal_variables)
        return eval(self.code, namespace)
        
    def get_dimensions(self, variable_units):
        '''
        Return the dimensions of the expression by evaluating it in its
        namespace, replacing all internal variables with their units.
        
        Parameters
        ----------
        variable_units : dict
            A dictionary mapping variable names to their units.
        
        Notes
        -----
        The namespace has to be resolved using the
        `~brian2.equations.codestrings.CodeString.resolve` method first.
        
        Raises
        ------
        DimensionMismatchError
            If the expression uses inconsistent units.
        '''
        return get_dimensions(self.eval(variable_units))
    
    
    def check_units(self, unit, variable_units):
        '''
        Check whether the dimensions of the expression match the expected
        dimensions.
        
        Parameters
        ----------
        unit : `Unit` or 1
            The expected unit (or 1 for dimensionless).
        variable_units : dict
            A dictionary mapping internal variable names to their units.                 
        
        Notes
        -----
        The namespace has to be resolved using the
        `~brian2.equations.codestrings.CodeString.resolve` method first.
        
        Raises
        ------
        DimensionMismatchError
            If the expression uses inconsistent units or the resulting unit does
            not match the expected `unit`.
        '''
        expr_dimensions = self.get_dimensions(variable_units)
        expected_dimensions = get_dimensions(unit)
        if not expr_dimensions == expected_dimensions:
            raise DimensionMismatchError('Dimensions of expression does not '
                                         'match its definition',
                                         expr_dimensions, expected_dimensions)
    
    def split_stochastic(self):
        '''
        Split the expression into a stochastic and non-stochastic part.
        
        Splits the expression into a tuple of two `Expression` objects f and g,
        assuming an expression of the form ``f + g * xi``, where ``xi`` is the
        symbol for the random variable.
        
        Returns
        -------
        (f, g) : (`Expression`, `Expression`)
            A tuple of `Expression` objects, the first one containing the
            non-stochastic and the second one containing the stochastic part.
            If no ``xi`` symbol is present in the code string, a tuple
            ``(self, None)`` will be returned with the unchanged `Expression`
            object.
        '''
        s_expr = self._sympy_expr.expand()
        xi = sympy.Symbol('xi')
        if not xi in s_expr:
            return (self, None)
        
        f = sympy.Wild('f', exclude=[xi]) # non-stochastic part
        g = sympy.Wild('g', exclude=[xi]) # stochastic part
        matches = s_expr.match(f + g * xi)
        if matches is None:
            raise ValueError(('Expression "%s" cannot be separated into stochastic '
                             'and non-stochastic term') % self.code)
    
        f_expr = Expression(str(matches[f]), namespace=self.namespace.copy(),
                            exhaustive=True)
        g_expr = Expression(str(matches[g] * xi), namespace=self.namespace.copy(),
                            exhaustive=True)
        
        return (f_expr, g_expr)
