import ast

from brian2.utils.stringtools import deindent
from collections import namedtuple

__all__ = ["abstract_code_dependencies"]


def get_read_write_funcs(parsed_code):
    allids = set([])
    read = set([])
    write = set([])
    funcs = set([])
    for node in ast.walk(parsed_code):
        if node.__class__ is ast.Name:
            allids.add(node.id)
            if node.ctx.__class__ is ast.Store:
                write.add(node.id)
            elif node.ctx.__class__ is ast.Load:
                read.add(node.id)
            else:
                raise SyntaxError
        elif node.__class__ is ast.Call:
            funcs.add(node.func.id)

    read = read - funcs

    # check that there's no funky stuff going on with functions
    if funcs.intersection(write):
        raise SyntaxError("Cannot assign to functions in abstract code")

    return allids, read, write, funcs


def abstract_code_dependencies(code, known_vars=None, known_funcs=None):
    """
    Analyses identifiers used in abstract code blocks

    Parameters
    ----------

    code : str
        The abstract code block.
    known_vars : set
        The set of known variable names.
    known_funcs : set
        The set of known function names.

    Returns
    -------

    results : namedtuple with the following fields
        ``all``
            The set of all identifiers that appear in this code block,
            including functions.
        ``read``
            The set of values that are read, excluding functions.
        ``write``
            The set of all values that are written to.
        ``funcs``
            The set of all function names.
        ``known_all``
            The set of all identifiers that appear in this code block and
            are known.
        ``known_read``
            The set of known values that are read, excluding functions.
        ``known_write``
            The set of known values that are written to.
        ``known_funcs``
            The set of known functions that are used.
        ``unknown_read``
            The set of all unknown variables whose values are read. Equal
            to ``read-known_vars``.
        ``unknown_write``
            The set of all unknown variables written to. Equal to
            ``write-known_vars``.
        ``unknown_funcs``
            The set of all unknown function names, equal to
            ``funcs-known_funcs``.
        ``undefined_read``
            The set of all unknown variables whose values are read before they
            are written to. If this set is nonempty it usually indicates an
            error, since a variable that is read should either have been
            defined in the code block (in which case it will appear in
            ``newly_defined``) or already be known.
        ``newly_defined``
            The set of all variable names which are newly defined in this
            abstract code block.
    """
    if known_vars is None:
        known_vars = set([])
    if known_funcs is None:
        known_funcs = set([])
    if not isinstance(known_vars, set):
        known_vars = set(known_vars)
    if not isinstance(known_funcs, set):
        known_funcs = set(known_funcs)

    code = deindent(code, docstring=True)
    parsed_code = ast.parse(code, mode="exec")

    # Get the list of all variables that are read from and written to,
    # ignoring the order
    allids, read, write, funcs = get_read_write_funcs(parsed_code)

    # Now check if there are any values that are unknown and read before
    # they are written to
    defined = known_vars.copy()
    newly_defined = set([])
    undefined_read = set([])
    for line in parsed_code.body:
        _, cur_read, cur_write, _ = get_read_write_funcs(line)
        undef = cur_read - defined
        undefined_read |= undef
        newly_defined |= (cur_write - defined) - undefined_read
        defined |= cur_write

    # Return the results as a named tuple
    results = dict(
        all=allids,
        read=read,
        write=write,
        funcs=funcs,
        known_all=allids.intersection(known_vars.union(known_funcs)),
        known_read=read.intersection(known_vars),
        known_write=write.intersection(known_vars),
        known_funcs=funcs.intersection(known_funcs),
        unknown_read=read - known_vars,
        unknown_write=write - known_vars,
        unknown_funcs=funcs - known_funcs,
        undefined_read=undefined_read,
        newly_defined=newly_defined,
    )
    return namedtuple("AbstractCodeDependencies", list(results.keys()))(**results)
