'''
Module defining `CodeString`, a class for a string of code together with
information about its namespace. Only serves as a parent class, its subclasses
`Expression` and `Statements` are the ones that are actually used.
'''
import sympy

from brian2.units.fundamentalunits import fail_for_dimension_mismatch
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers
from brian2.utils.parsing import parse_to_sympy

__all__ = ['Expression', 'Statements']

logger = get_logger(__name__)


class CodeString(object):
    '''
    A class for representing "code strings", i.e. a single Python expression
    or a sequence of Python statements.
    
    Parameters
    ----------
    code : str
        The code string, may be an expression or a statement(s) (possibly
        multi-line).
        
    '''

    def __init__(self, code):

        # : The code string
        self.code = code

        # : Set of identifiers in the code string
        self.identifiers = set(get_identifiers(code))

    def __str__(self):
        return self.code

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.code)


class Statements(CodeString):
    '''
    Class for representing statements.

    Parameters
    ----------
    code : str
        The statement or statements. Several statements can be given as a
        multi-line string or separated by semicolons.
    
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
    Class for representing an expression.

    Parameters
    ----------
    code : str
        The expression. Note that the expression has to be written in a form
        that is parseable by sympy.
    '''

    def __init__(self, code):
        CodeString.__init__(self, code)

        # : The expression as a sympy object
        self.sympy_expr = parse_to_sympy(self.code)

    def check_units(self, unit, namespace):
        '''
        Determines whether the result of the expression has the expected unit.
        Evaluates the code in the given namespace, which should contain
        `Unit` objects with the appropriate units for all internal variables
        (state variables and special variables like ``t``).
        
        Parameters
        ----------
        unit : `Unit`
            The expected unit.
        namespace: dict
            The namespace, mapping all identifiers in the expression to values
            (possibly `Unit` objects).
        
        Raises
        ------
        DimensionMismatchError
            In case the dimensions of the evaluated expression do not match the
            expected dimensions.
        '''
        fail_for_dimension_mismatch(unit, eval(self.code, namespace))

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
        s_expr = self.sympy_expr.expand()
        xi = sympy.Symbol('xi')
        if not xi in s_expr:
            return (self, None)

        f = sympy.Wild('f', exclude=[xi])  # non-stochastic part
        g = sympy.Wild('g', exclude=[xi])  # stochastic part
        matches = s_expr.match(f + g * xi)
        if matches is None:
            raise ValueError(('Expression "%s" cannot be separated into stochastic '
                             'and non-stochastic term') % self.code)

        f_expr = Expression(str(matches[f]))
        g_expr = Expression(str(matches[g] * xi))

        return (f_expr, g_expr)

    def _repr_pretty_(self, p, cycle):
        '''
        Pretty printing for ipython.
        '''
        if cycle:
            raise AssertionError('Cyclical call of CodeString._repr_pretty')
        # Make use of sympy's pretty printing
        p.pretty(self.sympy_expr)
