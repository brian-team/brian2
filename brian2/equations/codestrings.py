'''
Module defining `CodeString`, a class for a string of code together with
information about its namespace. Only serves as a parent class, its subclasses
`Expression` and `Statements` are the ones that are actually used.
'''

import inspect

import sympy
import numpy as np

from .unitcheck import get_default_unit_namespace, SPECIAL_VARS

from brian2.units.fundamentalunits import get_dimensions, DimensionMismatchError
import brian2.units.unitsafefunctions as usf
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers
from brian2.utils.parsing import parse_to_sympy

__all__ = ['Expression', 'Statements']

logger = get_logger(__name__)

def get_default_numpy_namespace():
    '''
    Get the namespace of numpy functions/variables that is recognized by
    default. The namespace includes the constants :np:attr:`pi`,
    :np:attr:`e` and :np:attr:`inf` and the following functions:
    :np:func:`abs`, :np:func:`arccos`, :np:func:`arccosh`,
    :np:func:`arcsin`, :np:func:`arcsinh`, :np:func:`arctan`,
    :np:func:`arctanh`, :np:func:`ceil`, :np:func:`clip`,
    :np:func:`cos`, :np:func:`cosh`, :np:func:`exp`,
    :np:func:`floor`, :np:func:`log`, :np:func:`max`,
    :np:func:`mean`, :np:func:`min`, :np:func:`prod`,
    :np:func:`round`, :np:func:`sin`, :np:func:`sinh`,
    :np:func:`std`, :np:func:`sum`, :np:func:`tan`,
    :np:func:`tanh`, :np:func:`var`, :np:func:`where`
    
    Returns
    -------
    namespace : dict
        A dictionary mapping function/variable names to numpy objects or
        their unitsafe Brian counterparts.
    '''        
    # numpy constants
    namespace = {'pi': np.pi, 'e': np.e, 'inf': np.inf}
    
    # standard numpy functions
    numpy_funcs = [np.abs, np.floor, np.ceil, np.round, np.min, np.max,
                   np.mean, np.std, np.var, np.sum, np.prod, np.clip]
    namespace.update([(func.__name__, func) for func in numpy_funcs])
    
    # unitsafe replacements for numpy functions
    replacements = [usf.log, usf.exp, usf.sin, usf.cos, usf.tan, usf.sinh,
                    usf.cosh, usf.tanh, usf.arcsin, usf.arccos, usf.arctan,
                    usf.arcsinh, usf.arccosh, usf.arctanh, usf.where]
    namespace.update([(func.__name__, func) for func in replacements])
    
    return namespace
    

def _conflict_warning(message, resolutions, the_logger):
    '''
    A little helper functions to generate warnings for logging. Specific
    to the `CodeString.resolve` method and should only be used by it.
    
    Parameters
    ----------
    message : str
        The first part of the warning message.
    resolutions : list of str
        A list of (namespace, object) tuples.
    the_logger : `BrianLogger`
        The logger object.
    '''
    if len(resolutions) == 0:
        # nothing to warn about
        return
    elif len(resolutions) == 1:
        second_part = ('but also refers to a variable in the %s namespace:'
                       ' %r') % (resolutions[0][0], resolutions[0][1])
    else:
        second_part = ('but also refers to a variable in the following '
                       'namespaces: %s') % (', '.join([r[0] for r in resolutions]))
    
    the_logger.warn(message + ' ' + second_part,
                    'CodeString.resolve.resolution_conflict')


class CodeString(object):
    '''
    A class for representing strings and corresponding namespaces.
    
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
        and globals. Ignored if `exhaustive` is ``True``.    
    
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
        
        self._namespaces = {}
        
        if namespace is None or not exhaustive:
            frame = inspect.stack()[level + 1][0]
            self._namespaces['locals'] = dict(frame.f_locals)
            self._namespaces['globals'] = dict(frame.f_globals)
        
        if namespace is not None:
            self._namespaces['user-defined'] = dict(namespace)

    code = property(lambda self: self._code,
                    doc='The code string.')
        
    identifiers = property(lambda self: self._identifiers,
                           doc='Set of identifiers in the code string.')

    namespaces = property(lambda self: self._namespaces,
                          doc='Namespaces that will be used for resolving the identifiers.')
    
    def replace_code(self, code):
        '''
        Return a new `CodeString` object with a new code string but the same
        namespace information. This function is for internal use, when a new
        `CodeString` object needs to be created from an existing one. 
        
        Parameters
        ----------
        code : str
            The new code string (see `CodeString` for more details).
        
        Returns
        -------
        codestring : `CodeString`
            A new `CodeString` object with the given `code` and the namespace
            information of the current object
        
        Notes
        -----
        The returned object will have the actual type of the object it is
        called on, e.g. calling on an `Expression` object will return an
        `Expression` object and not a generic `CodeString`.
        '''
        
        new_object = type(self)(code)
        new_object._namespaces = self._namespaces
        
        return new_object 

    def resolve(self, internal_variables):
        '''
        Determine the namespace, containing resolved references to externally
        defined variables and functions.

        Parameters
        ----------
        internal_variables : list of str
            A list of variables that should not be resolved.
        
        Returns
        -------
        namespace : dict
            A dictionary mapping names to objects, containing every identifier
            referenced in the code string, except for identifiers mentioned
            in `internal_variables`.
        
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

        unit_namespace = get_default_unit_namespace()
        numpy_namespace = get_default_numpy_namespace()
        
        resolved_namespace = {}
        for identifier in self.identifiers:
            # We save tuples of (namespace description, referred object) to
            # give meaningful warnings in case of duplicate definitions
            matches = []
            
            namespaces = self.namespaces.copy()
            namespaces.update({'units': unit_namespace,
                               'numpy': numpy_namespace})
            
            for description, namespace in namespaces.iteritems():
                if identifier in namespace:
                    matches.append((description, namespace[identifier]))            

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
                    # No match at all
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
                resolved_namespace[identifier] = matches[0][1]
                
        return resolved_namespace

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
        
        self._sympy_expr = parse_to_sympy(self.code)
         
    
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
        
        namespace = self.resolve(internal_variables).copy()
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
    
        f_expr = self.replace_code(str(matches[f]))
        g_expr = self.replace_code(str(matches[g] * xi))
        
        return (f_expr, g_expr)
    
    def _repr_pretty_(self, p, cycle):
        '''
        Pretty printing for ipython.
        '''
        if cycle:
            raise AssertionError('Cyclical call of CodeString._repr_pretty')
        # Make use of sympy's pretty printing
        p.pretty(self._sympy_expr)
