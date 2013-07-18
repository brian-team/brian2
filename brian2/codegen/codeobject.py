import functools

from brian2.core.specifiers import (ArrayVariable, Value,
                                    AttributeValue, Subexpression)
from .functions.base import Function
from brian2.core.preferences import brian_prefs, BrianPreference
from brian2.utils.logger import get_logger
from .translation import translate
from .runtime.targets import runtime_targets

__all__ = ['CodeObject',
           'create_codeobject',
           'get_codeobject_template',
           ]

logger = get_logger(__name__)


def get_default_codeobject_class():
    '''
    Returns the default `CodeObject` class from the preferences.
    '''
    codeobj_class = brian_prefs['codegen.target']
    if isinstance(codeobj_class, str):
        try:
            codeobj_class = runtime_targets[codeobj_class]
        except KeyError:
            raise ValueError("Unknown code generation target: %s, should be "
                             " one of %s"%(codeobj_class, runtime_targets.keys()))
    return codeobj_class


def prepare_namespace(namespace, specifiers):
    namespace = dict(namespace)
    # Add variables referring to the arrays
    arrays = []
    for value in specifiers.itervalues():
        if isinstance(value, ArrayVariable):
            arrays.append((value.arrayname, value.get_value()))
    namespace.update(arrays)

    return namespace


def create_codeobject(name, abstract_code, namespace, specifiers, template_name,
                      codeobj_class=None, indices=None, template_kwds=None):
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
    
    if codeobj_class is None:
        codeobj_class = get_default_codeobject_class()
        
    template = get_codeobject_template(template_name,
                                       codeobj_class=codeobj_class)

    namespace = prepare_namespace(namespace, specifiers)

    logger.debug(name + " abstract code:\n" + abstract_code)
    innercode, kwds = translate(abstract_code, specifiers, namespace,
                                brian_prefs['core.default_scalar_dtype'],
                                codeobj_class.language, indices)
    template_kwds.update(kwds)
    logger.debug(name + " inner code:\n" + str(innercode))
    code = template(innercode, **template_kwds)
    logger.debug(name + " code:\n" + str(code))

    specifiers.update(indices)
    codeobj = codeobj_class(code, namespace, specifiers)
    codeobj.compile()
    return codeobj


def get_codeobject_template(name, codeobj_class=None):
    '''
    Returns the `CodeObject` template ``name`` from the default or given class.
    '''
    if codeobj_class is None:
        codeobj_class = get_default_codeobject_class()
    return getattr(codeobj_class.templater, name)


class CodeObject(object):
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
    
    def __init__(self, code, namespace, specifiers):
        self.code = code
        self.compile_methods = self.get_compile_methods(specifiers)
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

    def get_compile_methods(self, specifiers):
        meths = []
        for var, spec in specifiers.items():
            if isinstance(spec, Function):
                meths.append(functools.partial(spec.on_compile,
                                               language=self.language,
                                               var=var))
        return meths

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

