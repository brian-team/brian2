'''
Base class for languages, gives the methods which should be overridden to
implement a new language.
'''
from brian2.core.variables import (ArrayVariable, AttributeVariable,
                                    Subexpression)
from brian2.utils.stringtools import get_identifiers

__all__ = ['Language']

class Language(object):
    '''
    Base class for all languages.
    
    See definition of methods below.
    
    TODO: more details here
    '''

    # Subclasses should override this
    language_id = ''

    def get_array_name(self, var, variables):
        '''
        Get a globally unique name for a `ArrayVariable`.

        Parameters
        ----------
        var : `ArrayVariable`
            The variable for which a name should be found.
        variables : dict
            The dictionary of variables, used to get a name for `var` if it
            doesn't have an owner.

        Returns
        -------
        name : str
            A uniqe name for `var`.
        '''

        if var.owner is None:
            # This should only happen for one-time use objects
            variable_dict = variables
            owner_name = ''
        else:
            variable_dict = var.owner.variables
            owner_name = var.owner.name + '_'
        orig_name = variable_dict.keys()[variable_dict.values().index(var)]

        return '_array_%s%s' % (owner_name, orig_name)

    def translate_expression(self, expr, namespace, codeobj_class):
        '''
        Translate the given expression string into a string in the target
        language, returns a string.
        '''
        raise NotImplementedError

    def translate_statement(self, statement, variables, codeobj_class):
        '''
        Translate a single line `Statement` into the target language, returns
        a string.
        '''
        raise NotImplementedError

    def translate_statement_sequence(self, statements, variables,
                                     variable_indices, iterate_all,
                                     codeobj_class):
        '''
        Translate a sequence of `Statement` into the target language, taking
        care to declare variables, etc. if necessary.
   
        Returns a tuple ``(code_lines, array_names, kwds)`` where ``code`` is
        list of the lines of code in the inner loop, ``array_names`` is a
        dictionary mapping variable names to the names of the underlying array
        (or a pointer to this array in the case of C code), and ``kwds`` is a
        dictionary of values that is made available to the template. Note that
        the content of ``array_names`` will also be added to the template
        keywords automatically.
        '''
        raise NotImplementedError

    def array_read_write(self, statements, variables, variable_indices):
        '''
        Helper function, gives the set of ArrayVariables that are read from and
        written to in the series of statements. Returns the pair read, write
        of sets of variable names.
        '''
        read = set()
        write = set()
        for stmt in statements:
            ids = set(get_identifiers(stmt.expr))
            # if the operation is inplace this counts as a read.
            if stmt.inplace:
                ids.add(stmt.var)
            read = read.union(ids)
            write.add(stmt.var)
        read = set(varname for varname, var in variables.items()
                   if isinstance(var, ArrayVariable) and varname in read)
        write = set(varname for varname, var in variables.items()
                    if isinstance(var, ArrayVariable) and varname in write)
        # Gather the indices stored as arrays (ignore _idx which is special)
        indices = set()
        indices |= set(variable_indices[varname] for varname in read
                       if variable_indices[varname] != '_idx'
                           and isinstance(variables[variable_indices[varname]],
                                          ArrayVariable))
        indices |= set(variable_indices[varname] for varname in write
                       if variable_indices[varname] != '_idx'
                           and isinstance(variables[variable_indices[varname]],
                                          ArrayVariable))
        # don't list arrays that are read explicitly and used as indices twice
        read -= indices
        return read, write, indices
