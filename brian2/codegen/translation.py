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

from brian2.core.variables import Variable, Subexpression, AuxiliaryVariable
from brian2.core.functions import Function
from brian2.utils.stringtools import (deindent, strip_empty_lines,
                                      get_identifiers, word_substitute)
from brian2.utils.topsort import topsort
from brian2.parsing.statements import parse_statement

from .statements import Statement


__all__ = ['make_statements', 'analyse_identifiers',
           'get_identifiers_recursively']

DEBUG = False


class LineInfo(object):
    '''
    A helper class, just used to store attributes.
    '''
    def __init__(self, **kwds):
        for k, v in kwds.iteritems():
            setattr(self, k, v)

    # TODO: This information should go somewhere else, I guess
STANDARD_IDENTIFIERS = set(['and', 'or', 'not', 'True', 'False'])


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
        variables = dict((k, Variable(unit=None, name=k,
                                      dtype=np.float64))
                         for k in known)

    known |= STANDARD_IDENTIFIERS
    stmts = make_statements(code, variables, np.float64)
    defined = set(stmt.var for stmt in stmts if stmt.op==':=')
    if recursive:
        if not isinstance(variables, collections.Mapping):
            raise TypeError('Have to specify a variables dictionary.')
        allids = get_identifiers_recursively(code, variables)
    else:
        allids = get_identifiers(code)
    dependent = allids.difference(defined, known)
    used_known = allids.intersection(known) - STANDARD_IDENTIFIERS
    return defined, used_known, dependent


def get_identifiers_recursively(expr, variables):
    '''
    Gets all the identifiers in a code, recursing down into subexpressions.
    '''
    identifiers = get_identifiers(expr)
    for name in set(identifiers):
        if name in variables and isinstance(variables[name], Subexpression):
            s_identifiers = get_identifiers_recursively(variables[name].expr,
                                                        variables)
            identifiers |= s_identifiers
    return identifiers


def make_statements(code, variables, dtype):
    '''
    Turn a series of abstract code statements into Statement objects, inferring
    whether each line is a set/declare operation, whether the variables are
    constant or not, and handling the cacheing of subexpressions. Returns a
    list of Statement objects. For arguments, see documentation for
    :func:`translate`.
    '''
    code = strip_empty_lines(deindent(code))
    lines = re.split(r'[;\n]', code)
    lines = [LineInfo(code=line) for line in lines if len(line)]
    if DEBUG:
        print 'INPUT CODE:'
        print code
    dtypes = dict((name, var.dtype) for name, var in variables.iteritems()
                  if not isinstance(var, Function))
    # we will do inference to work out which lines are := and which are =
    defined = set(k for k, v in variables.iteritems()
                  if not isinstance(v, AuxiliaryVariable))
    scalars = set(k for k,v in variables.iteritems()
                  if getattr(v, 'scalar', False))
    for line in lines:
        # parse statement into "var op expr"
        var, op, expr = parse_statement(line.code)
        if op=='=':
            if var not in defined:
                op = ':='
                defined.add(var)
                if var not in dtypes:
                    dtypes[var] = dtype
                # determine whether this is a scalar variable
                identifiers = get_identifiers_recursively(expr, variables)
                # In the following we assume that all unknown identifiers are
                # scalar constants -- this should cover numerical literals and
                # e.g. "True" or "inf".
                is_scalar = all((name in scalars) or not (name in defined)
                                for name in identifiers)
                if is_scalar:
                    scalars.add(var)

        statement = Statement(var, op, expr, dtypes[var], scalar=var in scalars)
        line.statement = statement
        # for each line will give the variable being written to
        line.write = var 
        # each line will give a set of variables which are read
        line.read = get_identifiers_recursively(expr, variables)

    # All writes to scalar variables must happen before writes to vector
    # variables
    scalar_write_done = False
    for line in lines:
        stmt = line.statement
        if stmt.op != ':=' and stmt.var in scalars and scalar_write_done:
            raise SyntaxError(('All writes to scalar variables in a code block '
                               'have to be made before writes to vector '
                               'variables. Illegal write to %s.') % line.write)
        elif not stmt.var in scalars:
            scalar_write_done = True

    if DEBUG:
        print 'PARSED STATEMENTS:'
        for line in lines:
            print line.statement, 'Read:'+str(line.read), 'Write:'+line.write
    
    # all variables which are written to at some point in the code block
    # used to determine whether they should be const or not
    all_write = set(line.write for line in lines)

    if DEBUG:
        print 'ALL WRITE:', all_write
        
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

    if DEBUG:
        print 'WILL READ/WRITE:'
        for line in lines:
            print line.statement, 'Read:'+str(line.will_read), 'Write:'+str(line.will_write)
        
    # generate cacheing statements for common subexpressions
    # cached subexpressions need to be recomputed whenever they are to be used
    # on the next line, and currently invalid (meaning that the current value
    # stored in the subexpression variable is no longer accurate because one
    # of the variables appearing in it has changed). All subexpressions start
    # as invalid, and are invalidated whenever one of the variables appearing
    # in the RHS changes value.
    subexpressions = dict((name, val) for name, val in variables.items() if isinstance(val, Subexpression))
    # sort subexpressions into an order so that subexpressions that don't depend
    # on other subexpressions are first
    subexpr_deps = dict((name, [dep for dep in subexpr.identifiers if dep in subexpressions]) for \
                                                            name, subexpr in subexpressions.items())
    sorted_subexpr_vars = topsort(subexpr_deps)

    if DEBUG:
        print 'SUBEXPRESSIONS:', subexpressions.keys()
    statements = []
    # all start as invalid
    valid = dict((name, False) for name in subexpressions.keys())
    # none are yet defined (or declared)
    subdefined = dict((name, False) for name in subexpressions.keys())
    for line in lines:
        stmt = line.statement
        read = line.read
        write = line.write
        will_read = line.will_read
        will_write = line.will_write
        # check that all subexpressions in expr are valid, and if not
        # add a definition/set its value, and set it to be valid
        # scan through in sorted order so that recursive subexpression dependencies
        # are handled in the right order
        for var in sorted_subexpr_vars:
            if var not in read:
                continue
            # if subexpression, and invalid
            if not valid.get(var, True): # all non-subexpressions are valid
                subexpression = subexpressions[var]
                # if already defined/declared
                if subdefined[var]:
                    op = '='
                    constant = False
                else:
                    op = ':='
                    subdefined[var] = True
                    dtypes[var] = variables[var].dtype
                    # set to constant only if we will not write to it again
                    constant = var not in will_write
                    # check all subvariables are not written to again as well
                    if constant:
                        ids = subexpression.identifiers
                        constant = all(v not in will_write for v in ids)
                valid[var] = True
                statement = Statement(var, op, subexpression.expr,
                                      variables[var].dtype, constant=constant,
                                      subexpression=True, scalar=var in scalars)
                statements.append(statement)
        var, op, expr = stmt.var, stmt.op, stmt.expr
        # invalidate any subexpressions including var, recursively
        # we do this by having a set of variables that are invalid that we
        # start with the changed var and increase by any subexpressions we
        # find that have a dependency on something in the invalid set. We
        # go through in sorted subexpression order so that the invalid set
        # is increased in the right order
        invalid = set([var])
        for subvar in sorted_subexpr_vars:
            spec = subexpressions[subvar]
            spec_ids = set(spec.identifiers)
            if spec_ids.intersection(invalid):
                valid[subvar] = False
                invalid.add(subvar)
        # constant only if we are declaring a new variable and we will not
        # write to it again
        constant = op==':=' and var not in will_write
        statement = Statement(var, op, expr, dtypes[var],
                              constant=constant, scalar=var in scalars)
        statements.append(statement)

    if DEBUG:
        print 'OUTPUT STATEMENTS:'
        for stmt in statements:
            print stmt

    return statements

