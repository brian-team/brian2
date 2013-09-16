'''
TODO: use preferences to get arguments to Language
'''
import numpy

from brian2.utils.stringtools import (deindent, stripped_deindented_lines,
                                      word_substitute)
from brian2.utils.logger import get_logger
from brian2.parsing.rendering import CPPNodeRenderer
from brian2.core.functions import (Function, FunctionImplementation,
                                   DEFAULT_FUNCTIONS)
from brian2.core.preferences import brian_prefs, BrianPreference
from brian2.core.variables import ArrayVariable

from .base import Language

logger = get_logger(__name__)

__all__ = ['CPPLanguage',
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


# Preferences
brian_prefs.register_preferences(
    'codegen.languages.cpp',
    'C++ codegen preferences',
    restrict_keyword = BrianPreference(
        default='__restrict__',
        docs='''
        The keyword used for the given compiler to declare pointers as restricted.
        
        This keyword is different on different compilers, the default is for gcc.
        ''',
        ),
    flush_denormals = BrianPreference(
        default=False,
        docs='''
        Adds code to flush denormals to zero.
        
        The code is gcc and architecture specific, so may not compile on all
        platforms. The code, for reference is::

            #define CSR_FLUSH_TO_ZERO         (1 << 15)
            unsigned csr = __builtin_ia32_stmxcsr();
            csr |= CSR_FLUSH_TO_ZERO;
            __builtin_ia32_ldmxcsr(csr);
            
        Found at `<http://stackoverflow.com/questions/2487653/avoiding-denormal-values-in-c>`_.
        ''',
        ),
    )


class CPPLanguage(Language):
    '''
    C++ language
    
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

    def __init__(self, c_data_type=c_data_type):
        self.restrict = brian_prefs['codegen.languages.cpp.restrict_keyword'] + ' '
        self.flush_denormals = brian_prefs['codegen.languages.cpp.flush_denormals']
        self.c_data_type = c_data_type

    def translate_expression(self, expr, namespace, codeobj_class):
        for varname, var in namespace.iteritems():
            if isinstance(var, Function):
                impl_name = var.implementations[codeobj_class].name
                if impl_name is not None:
                    expr = word_substitute(expr, {varname: impl_name})
        return CPPNodeRenderer().render_expr(expr).strip()

    def translate_statement(self, statement, namespace, codeobj_class):
        var, op, expr = statement.var, statement.op, statement.expr
        if op == ':=':
            decl = self.c_data_type(statement.dtype) + ' '
            op = '='
            if statement.constant:
                decl = 'const ' + decl
        else:
            decl = ''
        return decl + var + ' ' + op + ' ' + self.translate_expression(expr,
                                                                       namespace,
                                                                       codeobj_class) + ';'

    def translate_statement_sequence(self, statements, variables, namespace,
                                     variable_indices, iterate_all,
                                     codeobj_class):

        # Note that C++ code does not care about the iterate_all argument -- it
        # always has to loop over the elements

        read, write = self.array_read_write(statements, variables)
        lines = []
        # read arrays
        for varname in read:
            index_var = variable_indices[varname]
            var = variables[varname]
            if varname not in write:
                line = 'const '
            else:
                line = ''
            line = line + self.c_data_type(var.dtype) + ' ' + varname + ' = '
            line = line + '_ptr' + var.arrayname + '[' + index_var + '];'
            lines.append(line)
        # simply declare variables that will be written but not read
        for varname in write:
            if varname not in read:
                var = variables[varname]
                line = self.c_data_type(var.dtype) + ' ' + varname + ';'
                lines.append(line)
        # the actual code
        lines.extend([self.translate_statement(stmt, namespace, codeobj_class)
                      for stmt in statements])
        # write arrays
        for varname in write:
            index_var = variable_indices[varname]
            var = variables[varname]
            line = '_ptr' + var.arrayname + '[' + index_var + '] = ' + varname + ';'
            lines.append(line)
        code = '\n'.join(lines)
        # set up the restricted pointers, these are used so that the compiler
        # knows there is no aliasing in the pointers, for optimisation
        lines = []
        # It is possible that several different variable names refer to the
        # same array. E.g. in gapjunction code, v_pre and v_post refer to the
        # same array if a group is connected to itself
        arraynames = set()
        for varname, var in variables.iteritems():
            if isinstance(var, ArrayVariable):
                arrayname = var.arrayname
                if not arrayname in arraynames:
                    line = self.c_data_type(var.dtype) + ' * ' + self.restrict + '_ptr' + arrayname + ' = ' + arrayname + ';'
                    lines.append(line)
                    arraynames.add(arrayname)
        pointers = '\n'.join(lines)
        
        # set up the functions
        user_functions = []
        support_code = ''
        hash_defines = ''
        for varname, variable in namespace.items():
            if isinstance(variable, Function):
                user_functions.append((varname, variable))
                speccode = variable.implementations[codeobj_class].code
                if speccode is not None:
                    support_code += '\n' + deindent(speccode.get('support_code', ''))
                    hash_defines += deindent(speccode.get('hashdefine_code', ''))
                # add the Python function with a leading '_python', if it
                # exists. This allows the function to make use of the Python
                # function via weave if necessary (e.g. in the case of randn)
                if not variable.pyfunc is None:
                    pyfunc_name = '_python_' + varname
                    if pyfunc_name in namespace:
                        logger.warn(('Namespace already contains function %s, '
                                     'not replacing it') % pyfunc_name)
                    else:
                        namespace[pyfunc_name] = variable.pyfunc

        
        # delete the user-defined functions from the namespace and add the
        # function namespaces (if any)
        for funcname, func in user_functions:
            del namespace[funcname]
            func_namespace = func.implementations[codeobj_class].namespace
            if func_namespace is not None:
                namespace.update(func_namespace)

        return (stripped_deindented_lines(code),
                {'pointers_lines': stripped_deindented_lines(pointers),
                 'support_code_lines': stripped_deindented_lines(support_code),
                 'hashdefine_lines': stripped_deindented_lines(hash_defines),
                 'denormals_code_lines': stripped_deindented_lines(self.denormals_to_zero_code()),
                 })

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

################################################################################
# Implement functions
################################################################################

# Functions that exist under the same name in C++
for func in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log',
             'log10', 'sqrt', 'ceil', 'floor']:
    DEFAULT_FUNCTIONS[func].implementations[CPPLanguage] = FunctionImplementation()

# Functions that need a name translation
for func, func_cpp in [('arcsin', 'asin'), ('arccos', 'acos'), ('arctan', 'atan'),
                       ('abs', 'fabs'), ('mod', 'fmod')]:
    DEFAULT_FUNCTIONS[func].implementations[CPPLanguage] = FunctionImplementation(func_cpp)

# Functions that need to be implemented specifically
randn_code = {'support_code': '''

    inline double _ranf()
    {
        return (double)rand()/RAND_MAX;
    }

    double _randn(const int vectorisation_idx)
    {
         double x1, x2, w;
         static double y1, y2;
         static bool need_values = true;
         if (need_values)
         {
             do {
                     x1 = 2.0 * _ranf() - 1.0;
                     x2 = 2.0 * _ranf() - 1.0;
                     w = x1 * x1 + x2 * x2;
             } while ( w >= 1.0 );

             w = sqrt( (-2.0 * log( w ) ) / w );
             y1 = x1 * w;
             y2 = x2 * w;

             need_values = false;
             return y1;
         } else
         {
            need_values = true;
            return y2;
         }
    }
        '''}
DEFAULT_FUNCTIONS['randn'].implementations[CPPLanguage] = FunctionImplementation('_randn',
                                                                           code=randn_code)

rand_code = {'support_code': '''
        double _rand(int vectorisation_idx)
        {
	        return (double)rand()/RAND_MAX;
        }
        '''}
DEFAULT_FUNCTIONS['rand'].implementations[CPPLanguage] = FunctionImplementation('_rand',
                                                                          code=rand_code)

clip_code = {'support_code': '''
        double _clip(const float value, const float a_min, const float a_max)
        {
	        if (value < a_min)
	            return a_min;
	        if (value > a_max)
	            return a_max;
	        return value;
	    }
        '''}
DEFAULT_FUNCTIONS['clip'].implementations[CPPLanguage] = FunctionImplementation('_clip',
                                                                          code=clip_code)

int_code = {'support_code':
        '''
        int int_(const bool value)
        {
	        return value ? 1 : 0;
        }
        '''}
DEFAULT_FUNCTIONS['int_'].implementations[CPPLanguage] = FunctionImplementation('int_',
                                                                          code=int_code)