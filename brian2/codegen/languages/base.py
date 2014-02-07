'''
Base class for languages, gives the methods which should be overridden to
implement a new language.
'''
from brian2.core.variables import ArrayVariable
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

    def get_array_name(self, var, access_data=True):
        '''
        Get a globally unique name for a `ArrayVariable`.

        Parameters
        ----------
        var : `ArrayVariable`
            The variable for which a name should be found.
        access_data : bool, optional
            For `DynamicArrayVariable` objects, specifying `True` here means the
            name for the underlying data is returned. If specifying `False`,
            the name of object itself is returned (e.g. to allow resizing).
        Returns
        -------
        name : str
            A uniqe name for `var`.
        '''
         # We have to do the import here to avoid circular import dependencies.
        from brian2.devices.device import get_device
        device = get_device()
        return device.get_array_name(var, access_data=access_data)

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
                                     codeobj_class, override_conditional_write=None):
        '''
        Translate a sequence of `Statement` into the target language, taking
        care to declare variables, etc. if necessary.
   
        Returns a tuple ``(code_lines, array_names, dynamic_array_names, kwds)``
        where ``code`` is list of the lines of code in the inner loop,
        ``array_names`` is a dictionary mapping variable names to the names of
        the underlying array (or a pointer to this array in the case of C code),
        ``dynamic_array_names`` is a dictionary mapping ``_object``+variable
        names to the corresponding dynamic array objects and ``kwds`` is a
        dictionary of values that is made available to the template. Note that
        the content of ``array_names`` will also be added to the template
        keywords automatically. The same goes for ``dynamic_array_names`` but
        note that the keys in this array start all with ``_object``.
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

    def get_conditional_write_vars(self, variables, override_conditional_write):
        '''
        Helper function, returns a dict of mappings ``(varname, condition_var_name)`` indicating that
        when ``varname`` is written to, it should only be when ``condition_var_name`` is ``True``.
        '''
        conditional_write_vars = {}
        if override_conditional_write is None:
            override_conditional_write = set([])
        else:
            override_conditional_write = set(override_conditional_write)
        for varname, var in variables.items():
            if getattr(var, 'conditional_write', None) is not None:
                cvar = var.conditional_write
                cname = cvar.name
                if cname not in override_conditional_write:
                    conditional_write_vars[varname] = cname
        return conditional_write_vars

    def arrays_helper(self, statements, variables, variable_indices, override_conditional_write):
        '''
        Combines the two helper functions `array_read_write` and `get_conditional_write_vars`, and updates the
        ``read`` set.
        '''
        read, write, indices = self.array_read_write(statements, variables, variable_indices)
        conditional_write_vars = self.get_conditional_write_vars(variables, override_conditional_write)
        read = read.union(set(conditional_write_vars.values()) |
                          set(conditional_write_vars.keys()))
        return read, write, indices, conditional_write_vars
