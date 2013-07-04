'''
TODO: restrict keyword optimisations
'''
import itertools
import os

import numpy

from brian2.utils.stringtools import deindent, stripped_deindented_lines
from brian2.codegen.functions.base import Function
from brian2.utils.logger import get_logger

from ..base import Language, CodeObject
from ..templates import LanguageTemplater
from ...ast_parser import CPPNodeRenderer

logger = get_logger(__name__)
try:
    from scipy import weave
except ImportError as ex:
    logger.warn('Importing scipy.weave failed: %s' % ex)
    weave = None

__all__ = ['CPPLanguage', 'CPPCodeObject',
           'c_data_type',
           ]


def c_data_type(dtype):
    '''
    Gives the C language specifier for numpy data types. For example,
    ``numpy.int32`` maps to ``int32_t`` in C.
    '''
    # this handles the case where int is specified, it will be int32 or int64
    # depending on platform
    if dtype is int:
        dtype = numpy.array([1]).dtype.type
    if dtype is float:
        dtype = numpy.array([1.]).dtype.type

    if dtype == numpy.float32:
        dtype = 'float'
    elif dtype == numpy.float64:
        dtype = 'double'
    elif dtype == numpy.int32:
        dtype = 'int32_t'
    elif dtype == numpy.int64:
        dtype = 'int64_t'
    elif dtype == numpy.uint16:
        dtype = 'uint16_t'
    elif dtype == numpy.uint32:
        dtype = 'uint32_t'
    elif dtype == numpy.bool_ or dtype is bool:
        dtype = 'bool'
    else:
        raise ValueError("dtype " + str(dtype) + " not known.")
    return dtype


class CPPLanguage(Language):
    '''
    Initialisation arguments:
    
    ``compiler``
        The distutils name of the compiler.
    ``extra_compile_args``
        Extra compilation arguments, e.g. for optimisation. Best performance is
        often gained by using
        ``extra_compile_args=['-O3', '-ffast-math', '-march=native']`` 
        however the ``'-march=native'`` is not compatible with all versions of
        gcc so is switched off by default.
    ``restrict``
        The keyword used for the given compiler to declare pointers as
        restricted (different on different compilers).
    ``flush_denormals``
        Adds code to flush denormals to zero, but the code is gcc and
        architecture specific, so may not compile on all platforms, therefore
        it is off by default. The code, for reference is::

            #define CSR_FLUSH_TO_ZERO         (1 << 15)
            unsigned csr = __builtin_ia32_stmxcsr();
            csr |= CSR_FLUSH_TO_ZERO;
            __builtin_ia32_ldmxcsr(csr);
            
        Found at `<http://stackoverflow.com/questions/2487653/avoiding-denormal-values-in-c>`_.
        
    C++ code templates should provide Jinja2 macros with the following names:
    
    ``main``
        The main loop.
    ``support_code``
        The support code (function definitions, etc.), compiled in a separate
        file.
        
    For user-defined functions, there are two keys to provide:
    
    ``support_code``
        The function definition which will be added to the support code.
    ``hashdefine_code``
        The ``#define`` code added to the main loop.
        
    See `TimedArray` for an example of these keys.
    '''

    language_id = 'cpp'

    templater = LanguageTemplater(os.path.join(os.path.split(__file__)[0],
                                               'templates'))

    def __init__(self, compiler='gcc', extra_compile_args=['-w', '-O3', '-ffast-math'],
                 restrict='__restrict__', flush_denormals=False):
        self.compiler = compiler
        self.extra_compile_args = extra_compile_args
        self.restrict = restrict + ' '
        self.flush_denormals = flush_denormals

    def translate_expression(self, expr):
        return CPPNodeRenderer().render_expr(expr).strip()

    def translate_statement(self, statement):
        var, op, expr = statement.var, statement.op, statement.expr
        if op == ':=':
            decl = c_data_type(statement.dtype) + ' '
            op = '='
            if statement.constant:
                decl = 'const ' + decl
        else:
            decl = ''
        return decl + var + ' ' + op + ' ' + self.translate_expression(expr) + ';'

    def translate_statement_sequence(self, statements, specifiers, namespace, indices):
        read, write = self.array_read_write(statements, specifiers)
        lines = []
        # read arrays
        for var in read:
            index_var = specifiers[var].index
            index_spec = indices[index_var]
            spec = specifiers[var]
            if var not in write:
                line = 'const '
            else:
                line = ''
            line = line + c_data_type(spec.dtype) + ' ' + var + ' = '
            line = line + '_ptr' + spec.arrayname + '[' + index_var + '];'
            lines.append(line)
        # simply declare variables that will be written but not read
        for var in write:
            if var not in read:
                spec = specifiers[var]
                line = c_data_type(spec.dtype) + ' ' + var + ';'
                lines.append(line)
        # the actual code
        lines.extend([self.translate_statement(stmt) for stmt in statements])
        # write arrays
        for var in write:
            index_var = specifiers[var].index
            index_spec = indices[index_var]
            spec = specifiers[var]
            line = '_ptr' + spec.arrayname + '[' + index_var + '] = ' + var + ';'
            lines.append(line)
        code = '\n'.join(lines)
        # set up the restricted pointers, these are used so that the compiler
        # knows there is no aliasing in the pointers, for optimisation
        lines = []
        for var in read.union(write):
            spec = specifiers[var]
            line = c_data_type(spec.dtype) + ' * ' + self.restrict + '_ptr' + spec.arrayname + ' = ' + spec.arrayname + ';'
            lines.append(line)
        pointers = '\n'.join(lines)
        
        # set up the functions
        user_functions = []
        support_code = ''
        hash_defines = ''
        for var, spec in itertools.chain(namespace.items(),
                                         specifiers.items()):
            if isinstance(spec, Function):
                user_functions.append(var)
                speccode = spec.code(self, var)
                support_code += '\n' + deindent(speccode['support_code'])
                hash_defines += deindent(speccode['hashdefine_code'])
                # add the Python function with a leading '_python', if it
                # exists. This allows the function to make use of the Python
                # function via weave if necessary (e.g. in the case of randn)
                if not spec.pyfunc is None:
                    pyfunc_name = '_python_' + var
                    if pyfunc_name in  namespace:
                        logger.warn(('Namespace already contains function %s, '
                                     'not replacing it') % pyfunc_name)
                    else:
                        namespace[pyfunc_name] = spec.pyfunc
        
        # delete the user-defined functions from the namespace
        for func in user_functions:
            del namespace[func]
        
        # return
        return (stripped_deindented_lines(code),
                {'pointers_lines': stripped_deindented_lines(pointers),
                 'support_code_lines': stripped_deindented_lines(support_code),
                 'hashdefine_lines': stripped_deindented_lines(hash_defines),
                 'denormals_code_lines': stripped_deindented_lines(self.denormals_to_zero_code()),
                 })

    def code_object(self, code, namespace, specifiers):
        return CPPCodeObject(code,
                             namespace,
                             specifiers,
                             compile_methods=self.compile_methods(namespace),
                             compiler=self.compiler,
                             extra_compile_args=self.extra_compile_args)

    def denormals_to_zero_code(self):
        if self.flush_denormals:
            return '''
            #define CSR_FLUSH_TO_ZERO         (1 << 15)
            unsigned csr = __builtin_ia32_stmxcsr();
            csr |= CSR_FLUSH_TO_ZERO;
            __builtin_ia32_ldmxcsr(csr);
            '''
        else:
            return ''


class CPPCodeObject(CodeObject):
    '''
    C++ code object
    
    The ``code`` should be a `~brian2.codegen.languages.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    def __init__(self, code, namespace, specifiers, compile_methods=[],
                 compiler='gcc', extra_compile_args=['-O3']):
        super(CPPCodeObject, self).__init__(code,
                                            namespace,
                                            specifiers,
                                            compile_methods=compile_methods)
        self.compiler = compiler
        self.extra_compile_args = extra_compile_args

    def run(self):
        return weave.inline(self.code.main, self.namespace.keys(),
                            local_dict=self.namespace,
                            support_code=self.code.support_code,
                            compiler=self.compiler,
                            extra_compile_args=self.extra_compile_args)
