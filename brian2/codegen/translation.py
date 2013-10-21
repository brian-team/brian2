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

from numpy import float64

from brian2.core.variables import Variable, Subexpression, AuxiliaryVariable
from brian2.utils.stringtools import (deindent, strip_empty_lines,
                                      get_identifiers, word_substitute)
from brian2.parsing.statements import parse_statement

from .statements import Statement


__all__ = ['translate', 'make_statements', 'analyse_identifiers',
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
        variables = dict((k, Variable(unit=None, value=1.0)) for k in known)

    known |= STANDARD_IDENTIFIERS
    stmts = make_statements(code, variables, float64)
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
            s_identifiers = get_identifiers_recursively(translate_subexpression(variables[name], variables),
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
    dtypes = dict((name, var.dtype) for name, var in variables.iteritems())
    # we will do inference to work out which lines are := and which are =
    defined = set(k for k, v in variables.iteritems()
                  if not isinstance(v, AuxiliaryVariable))

    for line in lines:
        # parse statement into "var op expr"
        var, op, expr = parse_statement(line.code)
        if op=='=' and var not in defined:
            op = ':='
            defined.add(var)
            if var not in dtypes:
                dtypes[var] = dtype
        statement = Statement(var, op, expr, dtypes[var])
        line.statement = statement
        # for each line will give the variable being written to
        line.write = var 
        # each line will give a set of variables which are read
        line.read = get_identifiers_recursively(expr, variables)
        
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
    #subexpressions = get_all_subexpressions()
    subexpressions = dict((name, val) for name, val in variables.items() if isinstance(val, Subexpression))

    subexpressions = translate_subexpressions(subexpressions, variables)

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
        for var in read:
            # if subexpression, and invalid
            if not valid.get(var, True): # all non-subexpressions are valid
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
                        ids = subexpressions[var].identifiers
                        constant = all(v not in will_write for v in ids)
                valid[var] = True
                statement = Statement(var, op, subexpressions[var].expr,
                                      variables[var].dtype, constant=constant,
                                      subexpression=True)
                statements.append(statement)
        var, op, expr = stmt.var, stmt.op, stmt.expr
        # invalidate any subexpressions including var
        for subvar, spec in subexpressions.items():
            if var in spec.identifiers:
                valid[subvar] = False
        # constant only if we are declaring a new variable and we will not
        # write to it again
        constant = op==':=' and var not in will_write
        statement = Statement(var, op, expr, dtypes[var],
                              constant=constant)
        statements.append(statement)

    if DEBUG:
        print 'OUTPUT STATEMENTS:'
        for stmt in statements:
            print stmt

    return statements


def translate_subexpression(subexpr, variables):
    substitutions = {}
    for name in get_identifiers(subexpr.expr):
        if name not in subexpr.group.variables:
            # Seems to be a name referring to an external variable,
            # nothing to do
            continue
        subexpr_var = subexpr.group.variables[name]
        if name in variables and variables[name] is subexpr_var:
            # Variable is available under the same name, nothing to do
            continue

        # The variable is not available under the same name, but maybe
        # under a different name (e.g. x_post instead of x)
        found_variable = False
        for varname, variable in variables.iteritems():
            if variable is subexpr_var:
                # We found it
                substitutions[name] = varname
                found_variable = True
                break
        if not found_variable:
            raise KeyError(('Variable %s, referred to by the subexpression '
                            '%s, is not available in this '
                            'context.') % (name, subexpr.name))
    new_expr = word_substitute(subexpr.expr, substitutions)
    return new_expr


def translate_subexpressions(subexpressions, variables):
    new_subexpressions = {}
    for subexpr_name, subexpr in subexpressions.iteritems():
        new_expr = translate_subexpression(subexpr, variables)
        new_subexpressions[subexpr_name] = Subexpression(subexpr.name,
                                                         subexpr.unit,
                                                         expr=new_expr,
                                                         group=subexpr.group,
                                                         dtype=subexpr.dtype,
                                                         is_bool=subexpr.is_bool)

    subexpressions.update(new_subexpressions)
    return subexpressions

def translate(code, variables, namespace, dtype, codeobj_class,
              variable_indices, iterate_all):
    '''
    Translates an abstract code block into the target language.

    TODO
    
    Returns a multi-line string.
    '''
    if isinstance(code, dict):
        statements = {}
        for ac_name, ac_code in code.iteritems():
            statements[ac_name] = make_statements(ac_code, variables, dtype)
    else:
        statements = make_statements(code, variables, dtype)
    language = codeobj_class.language
    return language.translate_statement_sequence(statements, variables,
                                                 namespace, variable_indices,
                                                 iterate_all, codeobj_class)
