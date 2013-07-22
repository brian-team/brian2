'''
Base class for languages, gives the methods which should be overridden to
implement a new language.
'''
import functools
import numpy

from brian2.core.preferences import brian_prefs
from brian2.core.specifiers import (ArrayVariable, Value, AttributeValue,
                                    Subexpression)
from brian2.utils.stringtools import get_identifiers
from brian2.utils.logger import get_logger

from ..functions import Function
from ..translation import translate

logger = get_logger(__name__)

__all__ = ['Language', 'CodeObject']

class Language(object):
    '''
    Base class for all languages.
    
    See definition of methods below.
    
    TODO: more details here
    '''

    # Subclasses should override this
    language_id = ''
    
    # Subclasses should define a templater attribute

    def translate_expression(self, expr):
        '''
        Translate the given expression string into a string in the target
        language, returns a string.
        '''
        raise NotImplementedError

    def translate_statement(self, statement):
        '''
        Translate a single line Statement into the target language, returns
        a string.
        '''
        raise NotImplementedError

    def translate_statement_sequence(self, statements, specifiers, namespace, indices):
        '''
        Translate a sequence of Statements into the target language, taking
        care to declare variables, etc. if necessary.
   
        Returns a pair ``(code_lines, kwds)`` where ``code`` is a list of the
        lines of code in the inner loop, and ``kwds`` is a dictionary of values
        that is made available to the template.
        '''
        raise NotImplementedError

    def create_codeobj(self, name, abstract_code, namespace, specifiers,
                       template, indices=None, template_kwds=None):
        '''
        The following arguments keywords are passed to the template:
        
        * code_lines coming from translation applied to abstract_code, a list
          of lines of code, given to the template as ``code_lines`` keyword.
        * ``template_kwds`` dict
        * ``kwds`` coming from `translate` function overwrite those in
          ``template_kwds`` (but you should ensure there are no name
          clashes.
        '''
        if indices is None:  # TODO: Do we ever create code without any index?
            indices = {}
        if template_kwds is None:
            template_kwds = dict()
        else:
            template_kwds = template_kwds.copy()

        namespace = self.prepare_namespace(namespace, specifiers)

        logger.debug(name + " abstract code:\n" + abstract_code)
        innercode, kwds = translate(abstract_code, specifiers, namespace,
                                    brian_prefs['core.default_scalar_dtype'],
                                    self, indices)
        template_kwds.update(kwds)
        logger.debug(name + " inner code:\n" + str(innercode))
        code = template(innercode, **template_kwds)
        logger.debug(name + " code:\n" + str(code))

        specifiers.update(indices)
        codeobj = self.code_object(code, namespace, specifiers)
        codeobj.compile()
        return codeobj

    def prepare_namespace(self, namespace, specifiers):
        namespace = dict(namespace)
        # Add variables referring to the arrays
        arrays = []
        for value in specifiers.itervalues():
            if isinstance(value, ArrayVariable):
                arrays.append((value.arrayname, value.get_value()))
        namespace.update(arrays)

        return namespace

    def code_object(self, code, namespace, specifiers):
        '''
        Return an executable code object from the given code string.
        '''
        raise NotImplementedError

    def compile_methods(self, specifiers):
        meths = []
        for var, spec in specifiers.items():
            if isinstance(spec, Function):
                meths.append(functools.partial(spec.on_compile, language=self,
                                               var=var))
        return meths

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

    @property
    def template_state_update(self):
        return self.templater.stateupdate
    
    @property
    def template_reset(self):
        return self.templater.reset

    @property
    def template_threshold(self):
        return self.templater.threshold

    @property
    def template_spikemonitor(self):
        return self.templater.spikemonitor

    @property
    def template_statemonitor(self):
        return self.templater.statemonitor

    @property
    def template_synapses(self):
        return self.templater.synapses

    @property
    def template_synapses_create(self):
        return self.templater.synapses_create

    @property
    def template_state_variable_indexing(self):
        return self.templater.state_variable_indexing

    @property
    def template_lumped_variable(self):
        return self.templater.lumped_variable


class CodeObject(object):
    '''
    Executable code object, returned by Language
    
    Code object is initialised by Language object, typically just by doing
    ``CodeObject(code)``. The ``code`` can either be a string or a
    `brian2.codegen.languages.templates.MultiTemplate`.
    
    After initialisation, the code is compiled with the given namespace
    using ``code.compile(namespace)``.
    
    Calling ``code(key1=val1, key2=val2)`` executes the code with the given
    variables inserted into the namespace.
    '''
    def __init__(self, code, namespace, specifiers, compile_methods=[]):
        self.code = code
        self.compile_methods = compile_methods
        self.namespace = namespace
        self.specifiers = specifiers
        
        # Specifiers can refer to values that are either constant (e.g. dt)
        # or change every timestep (e.g. t). We add the values of the
        # constant specifiers here and add the names of non-constant specifiers
        # to a list
        
        # A list containing tuples of name and a function giving the value
        self.nonconstant_values = []
        
        for name, spec in self.specifiers.iteritems():   
            if isinstance(spec, Value):
                if isinstance(spec, AttributeValue):
                    self.nonconstant_values.append((name, spec.get_value))
                    if not spec.scalar:
                        self.nonconstant_values.append(('_num' + name,
                                                        spec.get_len))
                elif not isinstance(spec, Subexpression):
                    value = spec.get_value()
                    self.namespace[name] = value
                    # if it is a type that has a length, add a variable called
                    # '_num'+name with its length
                    if not spec.scalar:
                        self.namespace['_num' + name] = spec.get_len()


    def compile(self):
        for meth in self.compile_methods:
            meth(self.namespace)

    def __call__(self, **kwds):
        # update the values of the non-constant values in the namespace
        for name, func in self.nonconstant_values:
            self.namespace[name] = func()

        self.namespace.update(**kwds)

        return self.run()

    def run(self):
        '''
        Runs the code in the namespace.
        
        Returns
        -------
        return_value : dict
            A dictionary with the keys corresponding to the `output_variables`
            defined during the call of `Language.code_object`.
        '''
        raise NotImplementedError()
