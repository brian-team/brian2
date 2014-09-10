import itertools

import numpy as np

from brian2.utils.stringtools import word_substitute
from brian2.parsing.rendering import NumpyNodeRenderer
from brian2.core.functions import DEFAULT_FUNCTIONS, Function, SymbolicConstant
from brian2.core.variables import (ArrayVariable, Constant, AttributeVariable,
                                   DynamicArrayVariable, AuxiliaryVariable,
                                   get_dtype_str, Variable)

from .base import CodeGenerator
from .cpp_generator import c_data_type

from Cython.Build.Inline import unsafe_type, safe_type

__all__ = ['CythonCodeGenerator']


def cython_data_type(dtype):
    d = get_dtype_str(dtype)
    if d=='bool':
        d = 'uint8'
    return d


class CythonCodeGenerator(CodeGenerator):
    '''
    Cython code generator
    '''

    class_name = 'cython'

    def translate_expression(self, expr):
        # numpy version
        for varname, var in self.variables.iteritems():
            if isinstance(var, Function):
                impl_name = var.implementations[self.codeobj_class].name
                if impl_name is not None:
                    expr = word_substitute(expr, {varname: impl_name})
        return NumpyNodeRenderer().render_expr(expr, self.variables).strip()

    def translate_statement(self, statement):
        # TODO: optimisation, translate arithmetic to a sequence of inplace
        # operations like a=b+c -> add(b, c, a)
        var, op, expr, comment = (statement.var, statement.op,
                                  statement.expr, statement.comment)
        if op == ':=':
            op = '='
        code = var + ' ' + op + ' ' + self.translate_expression(expr)
        if len(comment):
            code += ' # ' + comment
        return code
        
    def translate_one_statement_sequence(self, statements):
        variables = self.variables
        variable_indices = self.variable_indices
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # index and read arrays (index arrays first)
        for varname in itertools.chain(indices, read):
            var = variables[varname]
            index = variable_indices[varname]
            line = '{varname} = {arrayname}[{index}]'.format(varname=varname, arrayname=self.get_array_name(var),
                                                             index=index)
            lines.append(line)
        # the actual code
        created_vars = set([])
        for stmt in statements:
            if stmt.op==':=':
                created_vars.add(stmt.var)
            line = self.translate_statement(stmt)
            if stmt.var in conditional_write_vars:
                subs = {}
                condvar = conditional_write_vars[stmt.var]
                lines.append('if %s:' % condvar)
                lines.append('    '+line)
            lines.append(line)
        # write arrays
        for varname in write:
            index_var = self.variable_indices[varname]
            var = self.variables[varname]
            line = self.get_array_name(var, self.variables) + '[' + index_var + '] = ' + varname
            lines.append(line)

        return lines

    def translate_statement_sequence(self, statements):
        from brian2.devices.device import get_device
        from brian2.codegen.runtime.weave_rt.weave_rt import weave_data_type
        device = get_device()
        # load variables from namespace
        load_namespace = []
        support_code = []
        handled_pointers = set()
        for varname, var in self.variables.iteritems():
            if isinstance(var, Variable):
                if var.dynamic:                
                    load_namespace.append('%s = _namespace["%s"]' % (self.get_array_name(var, False),
                                                                     self.get_array_name(var, False)))
                
                if not var.scalar:
                    # This is the "true" array name, not the restricted pointer.
                    array_name = device.get_array_name(var)
                    pointer_name = self.get_array_name(var)
                    if pointer_name in handled_pointers:
                        continue
                    if getattr(var, 'dimensions', 1) > 1:
                        continue  # multidimensional (dynamic) arrays have to be treated differently
                    newlines = [
                        "cdef _numpy.ndarray[{dtype_str_t}, ndim=1, mode='c'] _buf_{array_name} = _numpy.ascontiguousarray(_namespace['{array_name}'], dtype=_numpy.{dtype_str})",
                        "cdef {dtype} * {array_name} = <{dtype} *> _buf_{array_name}.data",
                        "cdef int _num{array_name} = len(_namespace['{array_name}'])",
                        ]
                    for line in newlines:
                        line = line.format(dtype=weave_data_type(var.dtype),
                                           pointer_name=pointer_name, array_name=array_name,
                                           varname=varname, dtype_str=var.dtype.__name__,
                                           dtype_str_t=('_numpy.'+var.dtype.__name__+'_t' if var.dtype.__name__!='bool' else '_numpy.uint8_t, cast=True'),
                                           )
                        load_namespace.append(line)
                    handled_pointers.add(pointer_name)
                elif isinstance(var, AttributeVariable):
                    val = getattr(var.obj, var.attribute)
                    dtype_name = unsafe_type(val)
                    dtype_name = dtype_name.replace('numpy.', '_numpy.')
                    line = 'cdef {dtype} {varname} = _namespace["{varname}"]'.format(dtype=dtype_name, varname=varname)
                    load_namespace.append(line)
                    if isinstance(val, np.ndarray):
                        line = "cdef int _num{varname} = len(_namespace['{varname}'])".format(varname=varname)
                        load_namespace.append(line)
                elif var.constant:
                    dtype_name = unsafe_type(var.value)
                    line = 'cdef {dtype} {varname} = _namespace["{varname}"]'.format(dtype=dtype_name, varname=varname)
                    load_namespace.append(line)
                elif isinstance(var, AuxiliaryVariable):
                    pass
            elif isinstance(var, Function):
                func_impl = var.implementations[self.codeobj_class].get_code(self.owner)
                # Implementation can be None if the function is already
                # available in Cython (possibly under a different name)
                if func_impl is not None:
                    if isinstance(func_impl, basestring):
                        # Function is provided as Cython code
                        support_code.append(func_impl)
                    elif callable(func_impl):
                        self.variables[varname] = func_impl
                        line = '%s = _namespace["%s"]' % (varname, varname)
                        load_namespace.append(line)
                    else:
                        raise TypeError(('Provided function implementation '
                                         'for function %s is neither a string '
                                         'nor callable') % varname)
            else:
                # fallback to Python object
                print var
                for k, v in var.__dict__.iteritems():
                    print '   ', k, v
                load_namespace.append('%s = _namespace["%s"]' % (varname, varname))
                
                
##            print varname, var.__class__
#            if isinstance(var, DynamicArrayVariable):
#                load_namespace.append('%s = _namespace["%s"]' % (self.get_array_name(var, False),
#                                                                 self.get_array_name(var, False)))
#            
#            if isinstance(var, ArrayVariable):
#                # This is the "true" array name, not the restricted pointer.
#                array_name = device.get_array_name(var)
#                pointer_name = self.get_array_name(var)
#                if pointer_name in handled_pointers:
#                    continue
#                if getattr(var, 'dimensions', 1) > 1:
#                    continue  # multidimensional (dynamic) arrays have to be treated differently
#                newlines = [
#                    "cdef _numpy.ndarray[{dtype_str_t}, ndim=1, mode='c'] _buf_{array_name} = _numpy.ascontiguousarray(_namespace['{array_name}'], dtype=_numpy.{dtype_str})",
#                    "cdef {dtype} * {array_name} = <{dtype} *> _buf_{array_name}.data",
#                    "cdef int _num{array_name} = len(_namespace['{array_name}'])",
#                    ]
#                for line in newlines:
#                    line = line.format(dtype=weave_data_type(var.dtype),
#                                       pointer_name=pointer_name, array_name=array_name,
#                                       varname=varname, dtype_str=var.dtype.__name__,
#                                       dtype_str_t=('_numpy.'+var.dtype.__name__+'_t' if var.dtype.__name__!='bool' else '_numpy.uint8_t, cast=True'),
#                                       )
#                    load_namespace.append(line)
#                handled_pointers.add(pointer_name)
#            elif isinstance(var, (Constant, SymbolicConstant)):
#                dtype_name = unsafe_type(var.value)
#                line = 'cdef {dtype} {varname} = _namespace["{varname}"]'.format(dtype=dtype_name, varname=varname)
#                load_namespace.append(line)
#            elif isinstance(var, AttributeVariable):
#                val = getattr(var.obj, var.attribute)
#                dtype_name = unsafe_type(val)
#                dtype_name = dtype_name.replace('numpy.', '_numpy.')
#                line = 'cdef {dtype} {varname} = _namespace["{varname}"]'.format(dtype=dtype_name, varname=varname)
#                load_namespace.append(line)
#                if isinstance(val, np.ndarray):
#                    line = "cdef int _num{varname} = len(_namespace['{varname}'])".format(varname=varname)
#                    load_namespace.append(line)
#            elif isinstance(var, Function):
#                func_impl = var.implementations[self.codeobj_class].get_code(self.owner)
#                # Implementation can be None if the function is already
#                # available in Cython (possibly under a different name)
#                if func_impl is not None:
#                    if isinstance(func_impl, basestring):
#                        # Function is provided as Cython code
#                        support_code.append(func_impl)
#                    elif callable(func_impl):
#                        self.variables[varname] = func_impl
#                        line = '%s = _namespace["%s"]' % (varname, varname)
#                        load_namespace.append(line)
#                    else:
#                        raise TypeError(('Provided function implementation '
#                                         'for function %s is neither a string '
#                                         'nor callable') % varname)
#            elif isinstance(var, AuxiliaryVariable):
#                pass
#            else:
#                # fallback to Python object
#                print var
#                for k, v in var.__dict__.iteritems():
#                    print '   ', k, v
#                load_namespace.append('%s = _namespace["%s"]' % (varname, varname))


        load_namespace = '\n'.join(load_namespace)
        support_code = '\n'.join(support_code)
        # main scalar/vector code
        scalar_code = {}
        vector_code = {}
        for name, block in statements.iteritems():
            scalar_statements = [stmt for stmt in block if stmt.scalar]
            vector_statements = [stmt for stmt in block if not stmt.scalar]
            scalar_code[name] = self.translate_one_statement_sequence(scalar_statements)
            vector_code[name] = self.translate_one_statement_sequence(vector_statements)

        return scalar_code, vector_code, {'load_namespace': load_namespace,
                                          'support_code': support_code}

###############################################################################
# Implement functions
################################################################################
# Functions that exist under the same name in C++
for func in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log',
             'log10', 'sqrt', 'ceil', 'floor']:
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(CythonCodeGenerator,
                                                               code=None)

# Functions that need a name translation
for func, func_cpp in [('arcsin', 'asin'), ('arccos', 'acos'), ('arctan', 'atan'),
                       ('abs', 'fabs'), ('mod', 'fmod')]:
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(CythonCodeGenerator,
                                                               code=None,
                                                               name=func_cpp)

clip_code = '''
def clip(float value, float a_min, float a_max):
    if value < a_min:
        return a_min
    elif value > a_max:
        return a_max
    else:
        return value
'''
DEFAULT_FUNCTIONS['clip'].implementations.add_implementation(CythonCodeGenerator,
                                                             code=clip_code)
int_func = lambda value: np.int32(value)
DEFAULT_FUNCTIONS['int'].implementations.add_implementation(CythonCodeGenerator,
                                                            code=int_func)
ceil_func = lambda value: np.int32(np.ceil(value))
DEFAULT_FUNCTIONS['ceil'].implementations.add_implementation(CythonCodeGenerator,
                                                            code=ceil_func)
floor_func = lambda value: np.int32(np.floor(value))
DEFAULT_FUNCTIONS['floor'].implementations.add_implementation(CythonCodeGenerator,
                                                            code=floor_func)



# TEMPORARY: Random number generation - NOT efficient
def randn_func(vectorisation_idx):
    return np.random.randn()

def rand_func(vectorisation_idx):
    return np.random.rand()

DEFAULT_FUNCTIONS['randn'].implementations.add_implementation(CythonCodeGenerator,
                                                              code=randn_func)

DEFAULT_FUNCTIONS['rand'].implementations.add_implementation(CythonCodeGenerator,
                                                             code=rand_func)
