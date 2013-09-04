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

    def translate_expression(self, expr, namespace):
        for varname, var in namespace.iteritems():
            if isinstance(var, Function):
                impl_name = var.implementations[self.language_id].name
                if varname != impl_name:
                    expr = word_substitute(expr, {varname: impl_name})
        return CPPNodeRenderer().render_expr(expr).strip()

    def translate_statement(self, statement, namespace):
        var, op, expr = statement.var, statement.op, statement.expr
        if op == ':=':
            decl = self.c_data_type(statement.dtype) + ' '
            op = '='
            if statement.constant:
                decl = 'const ' + decl
        else:
            decl = ''
        return decl + var + ' ' + op + ' ' + self.translate_expression(expr, namespace) + ';'

    def translate_statement_sequence(self, statements, variables, namespace,
                                     variable_indices, iterate_all):

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
        lines.extend([self.translate_statement(stmt, namespace) for stmt in statements])
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
                user_functions.append(varname)
                speccode = variable.code(self.language_id)
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
    DEFAULT_FUNCTIONS[func].implementations['cpp'] = FunctionImplementation(func)

# Functions that need a name translation
for func, func_cpp in [('arcsin', 'asin'), ('arccos', 'acos'), ('arctan', 'atan'),
                       ('abs', 'fabs'), ('mod', 'fmod')]:
    DEFAULT_FUNCTIONS[func].implementations['cpp'] = FunctionImplementation(func_cpp)

# Functions that need to be implemented specifically
randn_code = {'support_code': '''
        #define BUFFER_SIZE 1024
        // A randn() function that returns a single random number. Internally
        // it asks numpy's randn function for N (e.g. the number of neurons)
        // random numbers at a time and then returns one number from this
        // buffer.
        // It needs a reference to the numpy_randn object (the original numpy
        // function), because this is otherwise only available in
        // compiled_function (where is is automatically handled by weave).
        //
        double _call_randn(py::object& numpy_randn) {
            static PyArrayObject *randn_buffer = NULL;
            static double *buf_pointer = NULL;
            static npy_int curbuffer = 0;
            if(curbuffer==0)
            {
                if(randn_buffer) Py_DECREF(randn_buffer);
                py::tuple args(1);
                args[0] = BUFFER_SIZE;
                randn_buffer = (PyArrayObject *)PyArray_FromAny(numpy_randn.call(args), NULL, 1, 1, 0, NULL);
                buf_pointer = (double*)PyArray_GETPTR1(randn_buffer, 0);
            }
            double number = buf_pointer[curbuffer];
            curbuffer = curbuffer+1;
            if (curbuffer == BUFFER_SIZE)
                // This seems to be safer then using (curbuffer + 1) % BUFFER_SIZE, we might run into
                // an integer overflow for big networks, otherwise.
                curbuffer = 0;
            return number;
        }
        ''', 'hashdefine_code': '''
        #define _randn(_vectorisation_idx) _call_randn(_python_randn)
        '''}
DEFAULT_FUNCTIONS['randn'].implementations['cpp'] = FunctionImplementation('_randn',
                                                                           code=randn_code)

rand_code = {'support_code': '''
        double _rand(int vectorisation_idx)
        {
	        return (double)rand()/RAND_MAX;
        }
        '''}
DEFAULT_FUNCTIONS['rand'].implementations['cpp'] = FunctionImplementation('_rand',
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
DEFAULT_FUNCTIONS['clip'].implementations['cpp'] = FunctionImplementation('_clip',
                                                                          code=clip_code)

int_code = {'support_code':
        '''
        int int_(const bool value)
        {
	        return value ? 1 : 0;
        }
        '''}
DEFAULT_FUNCTIONS['int_'].implementations['cpp'] = FunctionImplementation('int_',
                                                                          code=int_code)