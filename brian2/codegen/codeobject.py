import functools
import weakref

from brian2.core.variables import (ArrayVariable, Variable,
                                    AttributeVariable, Subexpression,
                                    StochasticVariable)
from .functions.base import Function
from brian2.core.preferences import brian_prefs
from brian2.core.names import Nameable, find_name
from brian2.core.base import Updater
from brian2.utils.logger import get_logger
from .translation import translate
from .runtime.targets import runtime_targets

__all__ = ['CodeObject',
           'create_codeobject',
           'CodeObjectUpdater',
           ]

logger = get_logger(__name__)


def prepare_namespace(namespace, variables):
    namespace = dict(namespace)
    # Add variables referring to the arrays
    arrays = []
    for value in variables.itervalues():
        if isinstance(value, ArrayVariable):
            arrays.append((value.arrayname, value.get_value()))
    namespace.update(arrays)

    return namespace


def create_codeobject(name, abstract_code, namespace, variables, template_name,
                      indices, variable_indices, codeobj_class,
                      template_kwds=None):
    '''
    The following arguments keywords are passed to the template:
    
    * code_lines coming from translation applied to abstract_code, a list
      of lines of code, given to the template as ``code_lines`` keyword.
    * ``template_kwds`` dict
    * ``kwds`` coming from `translate` function overwrite those in
      ``template_kwds`` (but you should ensure there are no name
      clashes.
    '''

    if template_kwds is None:
        template_kwds = dict()
    else:
        template_kwds = template_kwds.copy()

    template = getattr(codeobj_class.templater, template_name)

    namespace = prepare_namespace(namespace, variables)

    logger.debug(name + " abstract code:\n" + abstract_code)
    iterate_all = template.iterate_all
    if isinstance(abstract_code, dict):
        snippet = {}
        kwds = {}
        for ac_name, ac in abstract_code.iteritems():
            snip, snip_kwds = translate(ac, variables, namespace,
                                        dtype=brian_prefs['core.default_scalar_dtype'],
                                        language=codeobj_class.language,
                                        variable_indices=variable_indices,
                                        iterate_all=iterate_all)
            snippet[ac_name] = snip
            for k, v in snip_kwds:
                kwds[ac_name+'_'+k] = v
            
    else:
        snippet, kwds = translate(abstract_code, variables, namespace,
                                  dtype=brian_prefs['core.default_scalar_dtype'],
                                  language=codeobj_class.language,
                                  variable_indices=variable_indices,
                                  iterate_all=iterate_all)
    template_kwds.update(kwds)
    logger.debug(name + " snippet:\n" + str(snippet))
    
    name = find_name(name)

    variables.update(indices)
    
    code = template(snippet, variables=variables, codeobj_name=name, namespace=namespace, **template_kwds)
    logger.debug(name + " code:\n" + str(code))

    codeobj = codeobj_class(code, namespace, variables, name=name)
    codeobj.compile()
    return codeobj


class CodeObject(Nameable):
    '''
    Executable code object.
    
    The ``code`` can either be a string or a
    `brian2.codegen.templates.MultiTemplate`.
    
    After initialisation, the code is compiled with the given namespace
    using ``code.compile(namespace)``.
    
    Calling ``code(key1=val1, key2=val2)`` executes the code with the given
    variables inserted into the namespace.
    '''
    
    #: The `Language` used by this `CodeObject`
    language = None

    def __init__(self, code, namespace, variables, name='codeobject*'):
        Nameable.__init__(self, name=name)
        self.code = code
        self.compile_methods = self.get_compile_methods(variables)
        self.namespace = namespace
        self.variables = variables

        self.variables_to_namespace()

    def variables_to_namespace(self):
        '''
        Add the values from the variables dictionary to the namespace.
        This should involve calling the `Variable.get_value` methods and
        possibly take track of variables that need to be updated at every
        timestep (see `update_namespace`).
        '''
        raise NotImplementedError()

    def update_namespace(self):
        '''
        Update the namespace for this timestep. Should only deal with variables
        where *the reference* changes every timestep, i.e. where the current
        reference in `namespace` is not correct.
        '''
        pass

    def get_compile_methods(self, variables):
        meths = []
        for var, var in variables.items():
            if isinstance(var, Function):
                meths.append(functools.partial(var.on_compile,
                                               language=self.language,
                                               var=var))
        return meths

    def compile(self):
        for meth in self.compile_methods:
            meth(self.namespace)

    def __call__(self, **kwds):
        self.update_namespace()
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
    
    def get_updater(self):
        '''
        Returns a `CodeObjectUpdater` that updates this `CodeObject`
        '''
        return CodeObjectUpdater(self)


class CodeObjectUpdater(Updater):
    '''
    Used to update ``CodeObject``.
    '''
    def run(self):
        self.owner()
