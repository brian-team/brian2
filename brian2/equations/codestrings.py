'''
Module defining `CodeString`, a class for a string of code together with
information about its namespace. Only serves as a parent class, its subclasses
`Expression` and `Statements` are the ones that are actually used.
'''
import re
import string
from collections.abc import Hashable
from typing import Sequence
import numbers

import sympy
import numpy as np

from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers
from brian2.utils.topsort import topsort
from brian2.parsing.sympytools import str_to_sympy, sympy_to_str

__all__ = ['Expression', 'Statements']

logger = get_logger(__name__)


class CodeString(Hashable):
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
        self._code = code.strip()

        # : Set of identifiers in the code string
        self.identifiers = get_identifiers(code)
        self.template_identifiers = get_identifiers(code, template=True)

    code = property(lambda self: self._code,
                    doc='The code string')

    def __str__(self):
        return self.code

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.code)

    def __eq__(self, other):
        if not isinstance(other, CodeString):
            return NotImplemented
        return self.code == other.code

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.code)


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


class Default(dict):
    def __missing__(self, key):
        return f'{{{key}}}'


class Expression(CodeString):
    '''
    Class for representing an expression.

    Parameters
    ----------
    code : str, optional
        The expression. Note that the expression has to be written in a form
        that is parseable by sympy. Alternatively, a sympy expression can be
        provided (in the ``sympy_expression`` argument).
    sympy_expression : sympy expression, optional
        A sympy expression. Alternatively, a plain string expression can be
        provided (in the ``code`` argument).
    '''

    def __init__(self, code=None, sympy_expression=None):
        if code is None and sympy_expression is None:
            raise TypeError('Have to provide either a string or a sympy expression')
        if code is not None and sympy_expression is not None:
            raise TypeError('Provide a string expression or a sympy expression, not both')

        if code is None:
            code = sympy_to_str(sympy_expression)
        elif '{' not in code:
            # Just try to convert it to a sympy expression to get syntax errors
            # for incorrect expressions
            str_to_sympy(code)
        super(Expression, self).__init__(code=code)

    stochastic_variables = property(lambda self: {variable for variable in self.identifiers
                                                  if variable =='xi' or variable.startswith('xi_')},
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

        stochastic_symbols = [sympy.Symbol(variable, real=True)
                              for variable in stochastic_variables]

        # Note that collect only works properly if the expression is expanded
        collected = str_to_sympy(self.code).expand().collect(stochastic_symbols,
                                                             evaluate=False)

        f_expr = None
        stochastic_expressions = {}
        for var, s_expr in collected.items():
            expr = Expression(sympy_expression=s_expr)
            if var == 1:
                if any(s_expr.has(s) for s in stochastic_symbols):
                    raise AssertionError(('Error when separating expression '
                                          '"%s" into stochastic and non-'
                                          'stochastic term: non-stochastic '
                                          'part was determined to be "%s" but '
                                          'contains a stochastic symbol)' % (self.code,
                                                                             s_expr)))
                f_expr = expr
            elif var in stochastic_symbols:
                stochastic_expressions[str(var)] = expr
            else:
                raise ValueError(('Expression "%s" cannot be separated into '
                                  'stochastic and non-stochastic '
                                  'term') % self.code)

        if f_expr is None:
            f_expr = Expression('0.0')

        return f_expr, stochastic_expressions

    def _repr_pretty_(self, p, cycle):
        '''
        Pretty printing for ipython.
        '''
        if cycle:
            raise AssertionError('Cyclical call of CodeString._repr_pretty')
        # Make use of sympy's pretty printing
        p.pretty(str_to_sympy(self.code))

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        return self.code == other.code

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.code)

    def _do_substitution(self, to_replace, replacement):
        # Replacements can be lists, deal with single replacements
        # as single-element lists
        replaced_name = False
        replaced_placeholder = False
        if not isinstance(replacement, Sequence) or isinstance(replacement, str):
            replacement = [replacement]
        replacement_strs = []
        for one_replacement in replacement:
            if isinstance(one_replacement, str):
                if any(c not in string.ascii_letters + '_{}'
                       for c in one_replacement):
                    # Check whether the replacement can be interpreted as an expression
                    try:
                        expr = Expression(one_replacement)
                        replacement_strs.append(expr.code)
                    except SyntaxError:
                        raise SyntaxError(f'Replacement \'{one_replacement}\' for'
                                          f'\'{to_replace}\' is neither a name nor a '
                                          f'valid expression.')
                else:
                    replacement_strs.append(one_replacement)
            elif isinstance(one_replacement, (numbers.Number, np.ndarray)):
                if not getattr(one_replacement, 'shape', ()) == ():
                    raise TypeError(f'Cannot replace variable \'{to_replace}\' with an '
                                    f'array of values.')
                replacement_strs.append(repr(one_replacement))
            elif isinstance(one_replacement, Expression):
                replacement_strs.append(one_replacement.code)
            else:
                raise TypeError(f'Cannot replace \'{to_replace}\' with an object of type '
                                f'\'{type(one_replacement)}\'.')

        if len(replacement_strs) == 1:
            replacement_str = replacement_strs[0]
            # Be careful if the string is more than just a name/number
            if any(c not in string.ascii_letters + string.digits + '_.{}'
                   for c in replacement_str):
                replacement_str = '(' + replacement_str + ')'
        else:
            replacement_str = '(' + (' + '.join(replacement_strs)) + ')'

        new_expr = self
        if to_replace in new_expr.identifiers:
            code = new_expr.code
            new_expr = Expression(re.sub(r'(?<!\w|{)' + to_replace + r'(?!\w|})',
                                         replacement_str, code))
            replaced_name = True
        if to_replace in new_expr.template_identifiers:
            code = new_expr.code
            new_expr = Expression(code.replace('{' + to_replace + '}',
                                               replacement_str))
            replaced_placeholder = True
        if not (replaced_name or replaced_placeholder):
            raise KeyError(f'Replacement argument \'{to_replace}\' does not correspond '
                           f'to any name or placeholder in the equations.')
        if replaced_name and replaced_placeholder:
            logger.warn(f'Replacement argument \'{to_replace}\' replaced both a name '
                        f'and a placeholder \'{{{to_replace}}}\'.',
                        name_suffix='ambiguous_replacement')
        return new_expr

    def __call__(self, **replacements):
        if len(replacements) == 0:
            return self

        # Figure out in which order elements should be substituted
        dependencies = {}
        for to_replace, replacement in replacements.items():
            if not isinstance(replacement, Sequence) or isinstance(replacement, str):
                replacement = [replacement]
            for one_replacement in replacement:
                dependencies[to_replace] = set()
                if not isinstance(one_replacement, (numbers.Number, np.ndarray, str, Expression)):
                    raise TypeError(f'Cannot use an object of type \'{type(one_replacement)}\''
                                    f'to replace \'{to_replace}\' in an expression.')
                if isinstance(one_replacement, Expression):
                    dependencies[to_replace] |= one_replacement.identifiers | one_replacement.template_identifiers
        # We only care about dependencies to values that are replaced at the same time
        for dep_key, deps in dependencies.items():
            dependencies[dep_key] = {d for d in deps if d in dependencies}

        replacements_in_order = topsort(dependencies)[::-1]
        expr = self
        for to_replace in replacements_in_order:
            replacement = replacements[to_replace]
            expr = expr._do_substitution(to_replace, replacement)
        return expr


def is_constant_over_dt(expression, variables, dt_value):
    '''
    Check whether an expression can be considered as constant over a time step.
    This is *not* the case when the expression either:

    1. contains the variable ``t`` (except as the argument of a function that
       can be considered as constant over a time step, e.g. a `TimedArray` with
       a dt equal to or greater than the dt used to evaluate this expression)
    2. refers to a stateful function such as ``rand()``.

    Parameters
    ----------
    expression : `sympy.Expr`
        The (sympy) expression to analyze
    variables : dict
        The variables dictionary.
    dt_value : float or None
        The length of a timestep (without units), can be ``None`` if the
        time step is not yet known.

    Returns
    -------
    is_constant : bool
        Whether the expression can be considered to be constant over a time
        step.
    '''
    t_symbol = sympy.Symbol('t', real=True, positive=True)
    if expression == t_symbol:
        return False  # The full expression is simply "t"
    func_name = str(expression.func)
    func_variable = variables.get(func_name, None)
    if func_variable is not None and not func_variable.stateless:
        return False
    for arg in expression.args:
        if arg == t_symbol and dt_value is not None:
            # We found "t" -- if it is not the only argument of a locally
            # constant function we bail out
            if not (func_variable is not None and
                        func_variable.is_locally_constant(dt_value)):
                return False
        else:
            if not is_constant_over_dt(arg, variables, dt_value):
                return False
    return True
