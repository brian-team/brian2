'''
Module defining `CodeString`, a class for a string of code together with
information about its namespace. Only serves as a parent class, its subclasses
`Expression` and `Statements` are the ones that are actually used.
'''
import sympy

from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers
from brian2.parsing.sympytools import str_to_sympy, sympy_to_str

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
        self.identifiers = get_identifiers(code)

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
        self.sympy_expr = str_to_sympy(self.code)

    stochastic_variables = property(lambda self: set([variable for variable in self.identifiers
                                                      if variable =='xi' or variable.startswith('xi_')]),
                                    doc='Stochastic variables in this expression')

    def split_stochastic(self):
        '''
        Split the expression into a stochastic and non-stochastic part.
        
        Splits the expression into a tuple of one `Expression` objects f (the
        non-stochastic part) and a dictionary mapping stochastic variables
        to `Expression` objects. For example, an expression of the form 
        ``f + g * xi_1 + h * xi_2`` would be returned as:
        ``(f, {'xi_1': g, 'xi_2': h})``
        Note that the `Expression` objects for the stochastic parts do not
        include the stochastic variable itself. 
        
        Returns
        -------
        (f, d) : (`Expression`, dict)
            A tuple of an `Expression` object and a dictionary, the first
            expression being the non-stochastic part of the equation and 
            the dictionary mapping stochastic variables (``xi`` or starting
            with ``xi_``) to `Expression` objects. If no stochastic variable
            is present in the code string, a tuple ``(self, None)`` will be
            returned with the unchanged `Expression` object.
        '''
        stochastic_variables = []
        for identifier in self.identifiers:
            if identifier == 'xi' or identifier.startswith('xi_'):
                stochastic_variables.append(identifier)
        
        # No stochastic variable
        if not len(stochastic_variables):
            return (self, None)
        
        s_expr = self.sympy_expr.expand()
        
        stochastic_symbols = [sympy.Symbol(variable, real=True)
                              for variable in stochastic_variables]

        f = sympy.Wild('f', exclude=stochastic_symbols)  # non-stochastic part
        match_objects = [sympy.Wild('w_'+variable, exclude=stochastic_symbols)
                         for variable in stochastic_variables]
        match_expression = f
        for symbol, match_object in zip(stochastic_symbols, match_objects):
            match_expression += match_object * symbol
        matches = s_expr.match(match_expression)
        
        if matches is None:
            raise ValueError(('Expression "%s" cannot be separated into stochastic '
                              'and non-stochastic term') % self.code)

        f_expr = Expression(sympy_to_str(matches[f]))
        stochastic_expressions = dict((variable, Expression(sympy_to_str(matches[match_object])))
                                        for (variable, match_object) in
                                        zip(stochastic_variables, match_objects))

        return (f_expr, stochastic_expressions)

    def _repr_pretty_(self, p, cycle):
        '''
        Pretty printing for ipython.
        '''
        if cycle:
            raise AssertionError('Cyclical call of CodeString._repr_pretty')
        # Make use of sympy's pretty printing
        p.pretty(self.sympy_expr)
