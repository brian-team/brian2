'''
Base class for languages, gives the methods which should be overridden to
implement a new language.
'''

from brian2.codegen.specifiers import ArrayVariable
from brian2.utils.stringtools import get_identifiers

__all__ = ['Language', 'CodeObject']

class Language(object):
    def translate_expression(self, expr):
        '''
        Translate the given expression string into a string in the target
        language.
        '''
        raise NotImplementedError
    
    def translate_statement(self,statement):
        '''
        Translate a single line Statement into the target language.
        '''
        raise NotImplementedError

    def translate_statement_sequence(self, statements, specifiers):
        '''
        Translate a sequence of Statements into the target language, taking
        care to declare variables, etc. if necessary.
        '''
        raise NotImplementedError
    
    def code_object(self, code):
        '''
        Return an executable code object from the given code string.
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
            # if the operation is inplace this counts as a read.
            if stmt.inplace:
                ids.add(stmt.var)
            read = read.union(ids)
            write.add(stmt.var)
        read = set(var for var, spec in specifiers.items() if isinstance(spec, ArrayVariable) and var in read)
        write = set(var for var, spec in specifiers.items() if isinstance(spec, ArrayVariable) and var in write)
        return read, write

    def template_iterate_all(self, index, size):
        '''
        Return a template where the variable ``index`` ranges from ``0:size``.
        Both ``index`` and ``size`` should be strings.
        '''
        raise NotImplementedError
    
    def template_iterate_index_array(self, index, array, size):
        '''
        Return a template where the variable ``index`` ranges through the
        values in ``array`` which is of length ``size``, each of these should
        be a string.
        '''
        raise NotImplementedError

    def template_state_update(self):
        '''
        Template for state updater code, by default just iterate over all neurons
        '''
        return self.template_iterate_all('_neuron_idx', '_num_neurons')
    
    def template_reset(self):
        '''
        Template for state updater code, by default just iterate over ``_spikes``
        '''
        return self.template_iterate_index_array('_neuron_idx', '_spikes', '_num_spikes')
    
    def template_threshold(self):
        '''
        Template for threshold code
        '''
        raise NotImplementedError

    def template_synapses(self):
        '''
        Template for synapses code
        '''
        raise NotImplementedError


class CodeObject(object):
    '''
    Executable code object, returned by Language
    
    Code object is initialised by Language object, typically just by doing
    ``CodeObject(code)``.
    
    After initialisation, the code is compiled with the given namespace
    using ``code.compile(namespace)``.
    
    Calling ``code(key1=val1, key2=val2)`` executes the code with the given
    variables inserted into the namespace.
    '''
    def __init__(self, code):
        self.code = code
    
    def compile(self, namespace):
        raise NotImplementedError
    
    def __call__(self, **kwds):
        raise NotImplementedError
