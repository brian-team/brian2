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
    
    def translate_expression(self, expr):
        '''
        Translate the given expression string into a string in the target
        language, returns a string.
        '''
        raise NotImplementedError

    def translate_statement(self, statement):
        '''
        Translate a single line `Statement` into the target language, returns
        a string.
        '''
        raise NotImplementedError

    def translate_statement_sequence(self, statements, variables, namespace,
                                     variable_indices, iterate_all):
        '''
        Translate a sequence of `Statement` into the target language, taking
        care to declare variables, etc. if necessary.
   
        Returns a pair ``(code_lines, kwds)`` where ``code`` is a list of the
        lines of code in the inner loop, and ``kwds`` is a dictionary of values
        that is made available to the template.
        '''
        raise NotImplementedError

    def array_read_write(self, statements, variables):
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
        return read, write
