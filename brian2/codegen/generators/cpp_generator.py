import itertools

import numpy

from brian2.utils.stringtools import (deindent, stripped_deindented_lines,
                                      word_substitute)
from brian2.utils.logger import get_logger
from brian2.parsing.rendering import CPPNodeRenderer
from brian2.core.functions import Function, DEFAULT_FUNCTIONS
from brian2.core.preferences import prefs, BrianPreference
from brian2.core.variables import ArrayVariable

from .base import CodeGenerator

logger = get_logger(__name__)

__all__ = ['CPPCodeGenerator',
           'c_data_type'
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
    elif dtype == numpy.int8:
        dtype = 'int8_t'
    elif dtype == numpy.int16:
        dtype = 'int16_t'
    elif dtype == numpy.int32:
        dtype = 'int32_t'
    elif dtype == numpy.int64:
        dtype = 'int64_t'
    elif dtype == numpy.uint16:
        dtype = 'uint16_t'
    elif dtype == numpy.uint32:
        dtype = 'uint32_t'
    elif dtype == numpy.uint64:
        dtype = 'uint64_t'
    elif dtype == numpy.bool_ or dtype is bool:
        dtype = 'bool'
    else:
        raise ValueError("dtype " + str(dtype) + " not known.")
    return dtype


# Preferences
prefs.register_preferences(
    'codegen.generators.cpp',
    'C++ codegen preferences',
    restrict_keyword = BrianPreference(
        default='__restrict',
        docs='''
        The keyword used for the given compiler to declare pointers as restricted.
        
        This keyword is different on different compilers, the default works for
        gcc and MSVS.
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


mod_support_code = ''
typestrs = ['unsigned char', 'char', 'unsigned short', 'short', 'unsigned int', 'int', 'unsigned long', 'long',
            'unsigned long long', 'long long', 'float', 'double', 'long double']
floattypestrs = ['float', 'double', 'long double']
for ix, xtype in enumerate(typestrs):
    for iy, ytype in enumerate(typestrs):
        hightype = typestrs[max(ix, iy)]
        if xtype in floattypestrs or ytype in floattypestrs:
            expr = 'fmod(fmod(x, y)+y, y)'
        else:
            expr = '((x%y)+y)%y'
        mod_support_code += '''
        inline {hightype} _brian_mod({xtype} ux, {ytype} uy)
        {{
            const {hightype} x = ({hightype})ux;
            const {hightype} y = ({hightype})uy;
            return {expr};
        }}
        '''.format(hightype=hightype, xtype=xtype, ytype=ytype, expr=expr)

_universal_support_code = deindent(mod_support_code)+'''
#ifdef _MSC_VER
#define _brian_pow(x, y) (pow((double)(x), (y)))
#else
#define _brian_pow(x, y) (pow((x), (y)))
#endif
'''


class CPPCodeGenerator(CodeGenerator):
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

    class_name = 'cpp'

    universal_support_code = _universal_support_code

    def __init__(self, *args, **kwds):
        super(CPPCodeGenerator, self).__init__(*args, **kwds)
        self.c_data_type = c_data_type

    @property
    def restrict(self):
        return prefs['codegen.generators.cpp.restrict_keyword'] + ' '

    @property
    def flush_denormals(self):
        return prefs['codegen.generators.cpp.flush_denormals']

    @staticmethod
    def get_array_name(var, access_data=True):
        # We have to do the import here to avoid circular import dependencies.
        from brian2.devices.device import get_device
        device = get_device()
        if access_data:
            return '_ptr' + device.get_array_name(var)
        else:
            return device.get_array_name(var, access_data=False)

    def translate_expression(self, expr):
        expr = word_substitute(expr, self.func_name_replacements)
        return CPPNodeRenderer().render_expr(expr).strip()

    def translate_statement(self, statement):
        var, op, expr, comment = (statement.var, statement.op,
                                  statement.expr, statement.comment)
        # For C++ we replace complex expressions involving boolean variables into a sequence of
        # if/then expressions with simpler expressions. This is provided by the optimise_statements
        # function.
        if statement.used_boolean_variables is not None and len(statement.used_boolean_variables):
            used_boolvars = statement.used_boolean_variables
            bool_simp = statement.boolean_simplified_expressions
            if op == ':=':
                # we have to declare the variable outside the if/then statement (which
                # unfortunately means we can't make it const but the optimisation is worth
                # it anyway).
                codelines = [self.c_data_type(statement.dtype) + ' ' + var + ';']
                op = '='
            else:
                codelines = []
            firstline = True
            # bool assigns is a sequence of (var, value) pairs giving the conditions under
            # which the simplified expression simp_expr holds
            for bool_assigns, simp_expr in bool_simp.iteritems():
                # generate a boolean expression like ``var1 && var2 && !var3``
                atomics = []
                for boolvar, boolval in bool_assigns:
                    if boolval:
                        atomics.append(boolvar)
                    else:
                        atomics.append('!'+boolvar)
                if firstline:
                    line = ''
                else:
                    line = 'else '
                # only need another if statement when we have more than one boolean variables
                if firstline or len(used_boolvars)>1:
                    line += 'if('+(' && '.join(atomics))+')'
                line += '\n    '
                line += var + ' ' + op + ' ' + self.translate_expression(simp_expr) + ';'
                codelines.append(line)
                firstline = False
            code = '\n'.join(codelines)
        else:
            if op == ':=':
                decl = self.c_data_type(statement.dtype) + ' '
                op = '='
                if statement.constant:
                    decl = 'const ' + decl
            else:
                decl = ''
            code = decl + var + ' ' + op + ' ' + self.translate_expression(expr) + ';'
        if len(comment):
            code += ' // ' + comment
        return code
    
    def translate_to_read_arrays(self, statements):
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # index and read arrays (index arrays first)
        for varname in itertools.chain(indices, read):
            index_var = self.variable_indices[varname]
            var = self.variables[varname]
            if varname not in write:
                line = 'const '
            else:
                line = ''
            line = line + self.c_data_type(var.dtype) + ' ' + varname + ' = '
            line = line + self.get_array_name(var, self.variables) + '[' + index_var + '];'
            lines.append(line)
        return lines

    def translate_to_declarations(self, statements):
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # simply declare variables that will be written but not read
        for varname in write:
            if varname not in read and varname not in indices:
                var = self.variables[varname]
                line = self.c_data_type(var.dtype) + ' ' + varname + ';'
                lines.append(line)
        return lines

    def translate_to_statements(self, statements):
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # the actual code
        for stmt in statements:
            line = self.translate_statement(stmt)
            if stmt.var in conditional_write_vars:
                subs = {}
                condvar = conditional_write_vars[stmt.var]
                lines.append('if(%s)' % condvar)
                lines.append('    '+line)
            else:
                lines.append(line)
        return lines

    def translate_to_write_arrays(self, statements):
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # write arrays
        for varname in write:
            index_var = self.variable_indices[varname]
            var = self.variables[varname]
            line = self.get_array_name(var, self.variables) + '[' + index_var + '] = ' + varname + ';'
            lines.append(line)
        return lines

    def translate_one_statement_sequence(self, statements, scalar=False):
        # This function is refactored into four functions which perform the
        # four necessary operations. It's done like this so that code
        # deriving from this class can overwrite specific parts.
        lines = []
        # index and read arrays (index arrays first)
        lines += self.translate_to_read_arrays(statements)
        # simply declare variables that will be written but not read
        lines += self.translate_to_declarations(statements)
        # the actual code
        lines += self.translate_to_statements(statements)
        # write arrays
        lines += self.translate_to_write_arrays(statements)
        code = '\n'.join(lines)                
        return stripped_deindented_lines(code)

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

    def _add_user_function(self, varname, variable):
        impl = variable.implementations[self.codeobj_class]
        support_code = []
        hash_defines = []
        pointers = []
        user_functions = [(varname, variable)]
        funccode = impl.get_code(self.owner)
        if isinstance(funccode, basestring):
            funccode = {'support_code': funccode}
        if funccode is not None:
            # To make namespace variables available to functions, we
            # create global variables and assign to them in the main
            # code
            func_namespace = impl.get_namespace(self.owner) or {}
            for ns_key, ns_value in func_namespace.iteritems():
                if hasattr(ns_value, 'dtype'):
                    if ns_value.shape == ():
                        raise NotImplementedError((
                        'Directly replace scalar values in the function '
                        'instead of providing them via the namespace'))
                    type_str = c_data_type(ns_value.dtype) + '*'
                else:  # e.g. a function
                    type_str = 'py::object'
                support_code.append('static {0} _namespace{1};'.format(type_str,
                                                                       ns_key))
                pointers.append('_namespace{0} = {1};'.format(ns_key, ns_key))
            support_code.append(deindent(funccode.get('support_code', '')))
            hash_defines.append(deindent(funccode.get('hashdefine_code', '')))

        dep_hash_defines = []
        dep_pointers = []
        dep_support_code = []
        if impl.dependencies is not None:
            for dep_name, dep in impl.dependencies.iteritems():
                self.variables[dep_name] = dep
                hd, ps, sc, uf = self._add_user_function(dep_name, dep)
                dep_hash_defines.extend(hd)
                dep_pointers.extend(ps)
                dep_support_code.extend(sc)
                user_functions.extend(uf)

        return (dep_hash_defines + hash_defines,
                dep_pointers + pointers,
                dep_support_code + support_code,
                user_functions)

    def determine_keywords(self):
        # set up the restricted pointers, these are used so that the compiler
        # knows there is no aliasing in the pointers, for optimisation
        pointers = []
        # It is possible that several different variable names refer to the
        # same array. E.g. in gapjunction code, v_pre and v_post refer to the
        # same array if a group is connected to itself
        handled_pointers = set()
        template_kwds = {}
        # Again, do the import here to avoid a circular dependency.
        from brian2.devices.device import get_device
        device = get_device()
        for varname, var in self.variables.iteritems():
            if isinstance(var, ArrayVariable):
                # This is the "true" array name, not the restricted pointer.
                array_name = device.get_array_name(var)
                pointer_name = self.get_array_name(var)
                if pointer_name in handled_pointers:
                    continue
                if getattr(var, 'dimensions', 1) > 1:
                    continue  # multidimensional (dynamic) arrays have to be treated differently
                restrict = self.restrict
                # turn off restricted pointers for scalars for safety
                if var.scalar:
                    restrict = ' '
                line = '{0}* {1} {2} = {3};'.format(self.c_data_type(var.dtype),
                                                    restrict,
                                                    pointer_name,
                                                    array_name)
                pointers.append(line)
                handled_pointers.add(pointer_name)

        # set up the functions
        user_functions = []
        support_code = []
        hash_defines = []
        for varname, variable in self.variables.items():
            if isinstance(variable, Function):
                hd, ps, sc, uf = self._add_user_function(varname, variable)
                user_functions.extend(uf)
                support_code.extend(sc)
                pointers.extend(ps)
                hash_defines.extend(hd)


        # delete the user-defined functions from the namespace and add the
        # function namespaces (if any)
        for funcname, func in user_functions:
            del self.variables[funcname]
            func_namespace = func.implementations[self.codeobj_class].get_namespace(self.owner)
            if func_namespace is not None:
                self.variables.update(func_namespace)

        support_code.append(self.universal_support_code)


        keywords = {'pointers_lines': stripped_deindented_lines('\n'.join(pointers)),
                    'support_code_lines': stripped_deindented_lines('\n'.join(support_code)),
                    'hashdefine_lines': stripped_deindented_lines('\n'.join(hash_defines)),
                    'denormals_code_lines': stripped_deindented_lines('\n'.join(self.denormals_to_zero_code())),
                    }
        keywords.update(template_kwds)
        return keywords

################################################################################
# Implement functions
################################################################################

# Functions that exist under the same name in C++
for func in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log',
             'log10', 'sqrt', 'ceil', 'floor']:
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(CPPCodeGenerator,
                                                               code=None)

# Functions that need a name translation
for func, func_cpp in [('arcsin', 'asin'), ('arccos', 'acos'), ('arctan', 'atan'),
                       ('int', 'int_')  # from stdint_compat.h
                       ]:
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(CPPCodeGenerator,
                                                               code=None,
                                                               name=func_cpp)

abs_code = '''
#define _brian_abs std::abs
'''
DEFAULT_FUNCTIONS['abs'].implementations.add_implementation(CPPCodeGenerator,
                                                            code=abs_code,
                                                            name='_brian_abs')


# Functions that need to be implemented specifically
randn_code = '''

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
        '''
DEFAULT_FUNCTIONS['randn'].implementations.add_implementation(CPPCodeGenerator,
                                                              code=randn_code,
                                                              name='_randn')

rand_code = '''
        inline double _rand(int vectorisation_idx)
        {
	        return (double)rand()/RAND_MAX;
        }
        '''
DEFAULT_FUNCTIONS['rand'].implementations.add_implementation(CPPCodeGenerator,
                                                             code=rand_code,
                                                             name='_rand')

clip_code = '''
        inline double _clip(const double value, const double a_min, const double a_max)
        {
	        if (value < a_min)
	            return a_min;
	        if (value > a_max)
	            return a_max;
	        return value;
	    }
        '''
DEFAULT_FUNCTIONS['clip'].implementations.add_implementation(CPPCodeGenerator,
                                                             code=clip_code,
                                                             name='_clip')

sign_code = '''
        template <typename T> inline int sign_(T val) {
            return (T(0) < val) - (val < T(0));
        }
        '''
DEFAULT_FUNCTIONS['sign'].implementations.add_implementation(CPPCodeGenerator,
                                                             code=sign_code,
                                                             name='sign_')
