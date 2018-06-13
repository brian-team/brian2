'''
This module translates a series of statements into a language-specific
syntactically correct code block that can be inserted into a template.

It infers whether or not a variable can be declared as
constant, etc. It should handle common subexpressions, and so forth.

The input information needed:

* The sequence of statements (a multiline string) in standard mathematical form
* The list of known variables, common subexpressions and functions, and for each
  variable whether or not it is a value or an array, and if an array what the
  dtype is.
* The dtype to use for newly created variables
* The language to translate to
'''
import re
import collections

import numpy as np
import sympy

from brian2.core.preferences import prefs
from brian2.core.variables import Variable, Subexpression, AuxiliaryVariable
from brian2.parsing.bast import brian_ast
from brian2.utils.caching import cached
from brian2.core.functions import Function
from brian2.utils.stringtools import (deindent, strip_empty_lines,
                                      get_identifiers)
from brian2.utils.topsort import topsort
from brian2.parsing.statements import parse_statement
from brian2.parsing.sympytools import str_to_sympy, sympy_to_str

from .statements import Statement
from .optimisation import optimise_statements

__all__ = ['analyse_identifiers', 'get_identifiers_recursively']


class LineInfo(object):
    '''
    A helper class, just used to store attributes.
    '''
    def __init__(self, **kwds):
        for k, v in kwds.iteritems():
            setattr(self, k, v)

    # TODO: This information should go somewhere else, I guess
STANDARD_IDENTIFIERS = {'and', 'or', 'not', 'True', 'False'}


def analyse_identifiers(code, variables, recursive=False):
    '''
    Analyses a code string (sequence of statements) to find all identifiers by type.
    
    In a given code block, some variable names (identifiers) must be given as inputs to the code
    block, and some are created by the code block. For example, the line::
    
        a = b+c
        
    This could mean to create a new variable a from b and c, or it could mean modify the existing
    value of a from b or c, depending on whether a was previously known.
    
    Parameters
    ----------
    code : str
        The code string, a sequence of statements one per line.
    variables : dict of `Variable`, set of names
        Specifiers for the model variables or a set of known names
    recursive : bool, optional
        Whether to recurse down into subexpressions (defaults to ``False``).
    
    Returns
    -------
    newly_defined : set
        A set of variables that are created by the code block.
    used_known : set
        A set of variables that are used and already known, a subset of the
        ``known`` parameter.
    unknown : set
        A set of variables which are used by the code block but not defined by
        it and not previously known. Should correspond to variables in the
        external namespace.
    '''
    if isinstance(variables, collections.Mapping):
        known = set(k for k, v in variables.iteritems()
                    if not isinstance(k, AuxiliaryVariable))
    else:
        known = set(variables)
        variables = dict((k, Variable(name=k, dtype=np.float64))
                         for k in known)

    known |= STANDARD_IDENTIFIERS
    scalar_stmts, vector_stmts = make_statements(code, variables, np.float64, optimise=False)
    stmts = scalar_stmts + vector_stmts
    defined = set(stmt.var for stmt in stmts if stmt.op == ':=')
    if len(stmts) == 0:
        allids = set()
    elif recursive:
        if not isinstance(variables, collections.Mapping):
            raise TypeError('Have to specify a variables dictionary.')
        allids = get_identifiers_recursively([stmt.expr for stmt in stmts],
                                             variables) | set([stmt.var
                                                               for stmt in stmts])
    else:
        allids = set.union(*[get_identifiers(stmt.expr)
                             for stmt in stmts]) | set([stmt.var for stmt in stmts])
    dependent = allids.difference(defined, known)
    used_known = allids.intersection(known) - STANDARD_IDENTIFIERS
    return defined, used_known, dependent


def get_identifiers_recursively(expressions, variables, include_numbers=False):
    '''
    Gets all the identifiers in a list of expressions, recursing down into
    subexpressions.

    Parameters
    ----------
    expressions : list of str
        List of expressions to check.
    variables : dict-like
        Dictionary of `Variable` objects
    include_numbers : bool, optional
        Whether to include number literals in the output. Defaults to ``False``.
    '''
    if len(expressions):
        identifiers = set.union(*[get_identifiers(expr, include_numbers=include_numbers)
                                  for expr in expressions])
    else:
        identifiers = set()
    for name in set(identifiers):
        if name in variables and isinstance(variables[name], Subexpression):
            s_identifiers = get_identifiers_recursively([variables[name].expr],
                                                        variables,
                                                        include_numbers=include_numbers)
            identifiers |= s_identifiers
    return identifiers


def is_scalar_expression(expr, variables):
    '''
    Whether the given expression is scalar.

    Parameters
    ----------
    expr : str
        The expression to check
    variables : dict-like
        `Variable` and `Function` object for all the identifiers used in `expr`

    Returns
    -------
    scalar : bool
        Whether `expr` is a scalar expression
    '''
    # determine whether this is a scalar variable
    identifiers = get_identifiers_recursively([expr], variables)
    # In the following we assume that all unknown identifiers are
    # scalar constants -- this should cover numerical literals and
    # e.g. "True" or "inf".
    return all(name not in variables or
               getattr(variables[name], 'scalar', False) or
               (isinstance(variables[name], Function) and variables[name].stateless)
               for name in identifiers)

@cached
def make_statements(code, variables, dtype, optimise=True, blockname=''):
    '''
    make_statements(code, variables, dtype, optimise=True, blockname='')

    Turn a series of abstract code statements into Statement objects, inferring
    whether each line is a set/declare operation, whether the variables are
    constant or not, and handling the cacheing of subexpressions.

    Parameters
    ----------
    code : str
        A (multi-line) string of statements.
    variables : dict-like
        A dictionary of with `Variable` and `Function` objects for every
        identifier used in the `code`.
    dtype : `dtype`
        The data type to use for temporary variables
    optimise : bool, optional
        Whether to optimise expressions, including
        pulling out loop invariant expressions and putting them in new
        scalar constants. Defaults to ``False``, since this function is also
        used just to in contexts where we are not interested by this kind of
        optimisation. For the main code generation stage, its value is set by
        the `codegen.loop_invariant_optimisations` preference.
    blockname : str, optional
        A name for the block (used to name intermediate variables to avoid
        name clashes when multiple blocks are used together)
    Returns
    -------
    scalar_statements, vector_statements : (list of `Statement`, list of `Statement`)
        Lists with statements that are to be executed once and statements that
        are to be executed once for every neuron/synapse/... (or in a vectorised
        way)

    Notes
    -----
    If ``optimise`` is ``True``, then the
    ``scalar_statements`` may include newly introduced scalar constants that
    have been identified as loop-invariant and have therefore been pulled out
    of the vector statements. The resulting statements will also use augmented
    assignments where possible, i.e. a statement such as ``w = w + 1`` will be
    replaced by ``w += 1``. Also, statements involving booleans will have
    additional information added to them (see `Statement` for details)
    describing how the statement can be reformulated as a sequence of if/then
    statements. Calls `~brian2.codegen.optimisation.optimise_statements`.
    '''
    code = strip_empty_lines(deindent(code))
    lines = re.split(r'[;\n]', code)
    lines = [LineInfo(code=line) for line in lines if len(line)]
    # Do a copy so we can add stuff without altering the original dict
    variables = dict(variables)
    # we will do inference to work out which lines are := and which are =
    defined = set(k for k, v in variables.iteritems()
                  if not isinstance(v, AuxiliaryVariable))
    for line in lines:
        statement = None
        # parse statement into "var op expr"
        var, op, expr, comment = parse_statement(line.code)
        if var in variables and isinstance(variables[var], Subexpression):
            raise SyntaxError("Illegal line '{line}' in abstract code. "
                              "Cannot write to subexpression "
                              "'{var}'.".format(line=line.code,
                                                var=var))
        if op == '=':
            if var not in defined:
                op = ':='
                defined.add(var)
                if var not in variables:
                    annotated_ast = brian_ast(expr, variables)
                    is_scalar = annotated_ast.scalar
                    if annotated_ast.dtype == 'boolean':
                        use_dtype = bool
                    elif annotated_ast.dtype == 'integer':
                        use_dtype = int
                    else:
                        use_dtype = dtype
                    new_var = AuxiliaryVariable(var, dtype=use_dtype,
                                                scalar=is_scalar)
                    variables[var] = new_var
            elif not variables[var].is_boolean:
                sympy_expr = str_to_sympy(expr, variables)
                sympy_var = sympy.Symbol(var, real=True)
                try:
                    collected = sympy.collect(sympy_expr, sympy_var,
                                              exact=True, evaluate=False)
                except AttributeError:
                    # If something goes wrong during collection, e.g. collect
                    # does not work for logical expressions
                    collected = {1: sympy_expr}

                if (len(collected) == 2 and
                        set(collected.keys()) == {1, sympy_var} and
                        collected[sympy_var] == 1):
                    # We can replace this statement by a += assignment
                    statement = Statement(var, '+=',
                                          sympy_to_str(collected[1]),
                                          comment,
                                          dtype=variables[var].dtype,
                                          scalar=variables[var].scalar)
                elif len(collected) == 1 and sympy_var in collected:
                    # We can replace this statement by a *= assignment
                    statement = Statement(var, '*=',
                                          sympy_to_str(collected[sympy_var]),
                                          comment,
                                          dtype=variables[var].dtype,
                                          scalar=variables[var].scalar)
        if statement is None:
            statement = Statement(var, op, expr, comment,
                                  dtype=variables[var].dtype,
                                  scalar=variables[var].scalar)

        line.statement = statement
        # for each line will give the variable being written to
        line.write = var 
        # each line will give a set of variables which are read
        line.read = get_identifiers_recursively([expr], variables)

    # All writes to scalar variables must happen before writes to vector
    # variables
    scalar_write_done = False
    for line in lines:
        stmt = line.statement
        if stmt.op != ':=' and variables[stmt.var].scalar and scalar_write_done:
            raise SyntaxError(('All writes to scalar variables in a code block '
                               'have to be made before writes to vector '
                               'variables. Illegal write to %s.') % line.write)
        elif not variables[stmt.var].scalar:
            scalar_write_done = True

    # all variables which are written to at some point in the code block
    # used to determine whether they should be const or not
    all_write = set(line.write for line in lines)

    # backwards compute whether or not variables will be read again
    # note that will_read for a line gives the set of variables it will read
    # on the current line or subsequent ones. will_write gives the set of
    # variables that will be written after the current line
    will_read = set()
    will_write = set()
    for line in lines[::-1]:
        will_read = will_read.union(line.read)
        line.will_read = will_read.copy()
        line.will_write = will_write.copy()
        will_write.add(line.write)

    subexpressions = dict((name, val) for name, val in variables.items() if isinstance(val, Subexpression))
    # sort subexpressions into an order so that subexpressions that don't depend
    # on other subexpressions are first
    subexpr_deps = dict((name, [dep for dep in subexpr.identifiers if dep in subexpressions]) for \
                                                            name, subexpr in subexpressions.items())
    sorted_subexpr_vars = topsort(subexpr_deps)

    statements = []

    # none are yet defined (or declared)
    subdefined = dict((name, None) for name in subexpressions.keys())
    for line in lines:
        stmt = line.statement
        read = line.read
        write = line.write
        will_read = line.will_read
        will_write = line.will_write
        # update/define all subexpressions needed by this statement
        for var in sorted_subexpr_vars:
            if var not in read:
                continue

            subexpression = subexpressions[var]
            # if already defined/declared
            if subdefined[var] == 'constant':
                continue
            elif subdefined[var] == 'variable':
                op = '='
                constant = False
            else:
                op = ':='
                # check if the referred variables ever change
                ids = subexpression.identifiers
                constant = all(v not in will_write for v in ids)
                subdefined[var] = 'constant' if constant else 'variable'

            statement = Statement(var, op, subexpression.expr, comment='',
                                  dtype=variables[var].dtype,
                                  constant=constant,
                                  subexpression=True,
                                  scalar=variables[var].scalar)
            statements.append(statement)

        var, op, expr, comment = stmt.var, stmt.op, stmt.expr, stmt.comment

        # constant only if we are declaring a new variable and we will not
        # write to it again
        constant = op == ':=' and var not in will_write
        statement = Statement(var, op, expr, comment,
                              dtype=variables[var].dtype,
                              constant=constant,
                              scalar=variables[var].scalar)
        statements.append(statement)

    scalar_statements = [s for s in statements if s.scalar]
    vector_statements = [s for s in statements if not s.scalar]

    if optimise and prefs.codegen.loop_invariant_optimisations:
        scalar_statements, vector_statements = optimise_statements(scalar_statements,
                                                                   vector_statements,
                                                                   variables,
                                                                   blockname=blockname)

    return scalar_statements, vector_statements
