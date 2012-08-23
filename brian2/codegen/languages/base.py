'''
Base class for languages, gives the methods which should be overridden to
implement a new language.
'''

from brian2.codegen.specifiers import ArrayVariable
from brian2.utils.stringtools import get_identifiers

__all__ = ['Language']

class Language(object):
    def translate_expression(self, expr):
        '''
        Translate the given expression string into a string in the target
        language.
        '''
        raise NotImplementedError
    
    def translate_statement(statement):
        '''
        Translate a single line Statement into the target language.
        '''
        raise NotImplementedError

    def translate_statement_sequence(self, statements, specifiers,
                                     index_var, index_spec):
        '''
        Translate a sequence of Statements into the target language, taking
        care to declare variables, etc. if necessary. The ``index_var`` and
        ``index_spec`` are the values which will be iterated over.
        '''
        raise NotImplementedError

    def array_read_write(self, statements, specifiers):
        '''
        Helper function, gives the set of ArrayVariables that are read from and
        written to in the series of statements. Returns the pair read, write
        of sets of variable names.
        '''
        read = set()
        write = set()
        for stmt in statements:
            ids = set(get_identifiers(stmt.expr))
            read = read.union(ids)
            write.add(stmt.var)
        read = set(var for var, spec in specifiers.items() if isinstance(spec, ArrayVariable) and var in read)
        write = set(var for var, spec in specifiers.items() if isinstance(spec, ArrayVariable) and var in write)
        return read, write
