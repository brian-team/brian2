import itertools

import numpy as np

from brian2.utils.stringtools import word_substitute, deindent
from brian2.parsing.rendering import NodeRenderer
from brian2.core.functions import DEFAULT_FUNCTIONS, Function, SymbolicConstant
from brian2.core.variables import (ArrayVariable, Constant, AttributeVariable,
                                   DynamicArrayVariable, AuxiliaryVariable,
                                   get_dtype_str, Variable, Subexpression)

from .base import CodeGenerator


__all__ = ['CythonCodeGenerator']


data_type_conversion_table = [
    # canonical         C++            Numpy
    ('float32',        'float',       'float32'),
    ('float64',        'double',      'float64'),
    ('int32',          'int32_t',     'int32'),
    ('int64',          'int64_t',     'int64'),
    ('bool',           'char',        'uint8'),
    ('uint8',          'char',        'uint8'),
    ]

cpp_dtype = dict((canonical, cpp) for canonical, cpp, np in data_type_conversion_table)
numpy_dtype = dict((canonical, np) for canonical, cpp, np in data_type_conversion_table)

def get_cpp_dtype(obj):
    return cpp_dtype[get_dtype_str(obj)]

def get_numpy_dtype(obj):
    return numpy_dtype[get_dtype_str(obj)]


class CythonNodeRenderer(NodeRenderer):
    def render_NameConstant(self, node):
        return {True: '1',
                False: '0'}.get(node.value, node.value)

    def render_Name(self, node):
        return {'True': '1',
                'False': '0'}.get(node.id, node.id)


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
        return CythonNodeRenderer().render_expr(expr, self.variables).strip()

    def translate_statement(self, statement):
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
            else:
                lines.append(line)
        # write arrays
        for varname in write:
            index_var = self.variable_indices[varname]
            var = self.variables[varname]
            line = self.get_array_name(var, self.variables) + '[' + index_var + '] = ' + varname
            lines.append(line)

        return lines

    def determine_keywords(self):
        from brian2.devices.device import get_device
        device = get_device()
        # load variables from namespace
        load_namespace = []
        support_code = []
        handled_pointers = set()
        for varname, var in self.variables.iteritems():
            if isinstance(var, AuxiliaryVariable):
                line = "cdef {dtype} {varname}".format(
                                dtype=get_cpp_dtype(var.dtype),
                                varname=varname)
                load_namespace.append(line)
            elif isinstance(var, AttributeVariable):
                val = getattr(var.obj, var.attribute)
                if isinstance(val, np.ndarray) and val.ndim:
                    line = "cdef _numpy.ndarray[{cpp_dtype}, ndim=1, mode='c'] {varname} = _namespace['{varname}']".format(
                        numpy_dtype=get_numpy_dtype(val), varname=varname,
                        cpp_dtype=get_cpp_dtype(val))
                else:
                    line = "cdef {cpp_dtype} {varname} = _namespace['{varname}']".format(
                        cpp_dtype=get_cpp_dtype(val), varname=varname)
                load_namespace.append(line)
                if isinstance(val, np.ndarray) and val.ndim:
                    line = "cdef int _num{varname} = len(_namespace['{varname}'])".format(varname=varname)
                    load_namespace.append(line)
            elif isinstance(var, Subexpression):
                dtype = get_cpp_dtype(var.dtype)
                line = "cdef {dtype} {varname}".format(dtype=dtype,
                                                       varname=varname)
                load_namespace.append(line)
            elif isinstance(var, Constant):
                dtype_name = get_cpp_dtype(var.value)
                line = 'cdef {dtype} {varname} = _namespace["{varname}"]'.format(dtype=dtype_name, varname=varname)
                load_namespace.append(line)
            elif isinstance(var, Variable):
                if var.dynamic:
                    load_namespace.append('%s = _namespace["%s"]' % (self.get_array_name(var, False),
                                                                     self.get_array_name(var, False)))

                # This is the "true" array name, not the restricted pointer.
                array_name = device.get_array_name(var)
                pointer_name = self.get_array_name(var)
                if pointer_name in handled_pointers:
                    continue
                if getattr(var, 'dimensions', 1) > 1:
                    continue  # multidimensional (dynamic) arrays have to be treated differently
                newlines = [
                    "cdef _numpy.ndarray[{cpp_dtype}, ndim=1, mode='c'] _buf_{array_name} = _namespace['{array_name}'].view(dtype=_numpy.{numpy_dtype})",
                    "cdef {cpp_dtype} * {array_name} = <{cpp_dtype} *> _buf_{array_name}.data",]

                if not var.scalar:
                    newlines += ["cdef int _num{array_name} = len(_namespace['{array_name}'])"]

                newlines += ["cdef {cpp_dtype} {varname}"]

                for line in newlines:
                    line = line.format(cpp_dtype=get_cpp_dtype(var.dtype),
                                       numpy_dtype=get_numpy_dtype(var.dtype),
                                       pointer_name=pointer_name,
                                       array_name=array_name,
                                       varname=varname,
                                       )
                    load_namespace.append(line)
                handled_pointers.add(pointer_name)

            elif isinstance(var, Function):
                func_impl = var.implementations[self.codeobj_class].get_code(self.owner)
                # Implementation can be None if the function is already
                # available in Cython (possibly under a different name)
                if func_impl is not None:
                    if isinstance(func_impl, basestring):
                        # Function is provided as Cython code
                        support_code.append(deindent(func_impl))
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

        load_namespace = '\n'.join(load_namespace)
        support_code = '\n'.join(support_code)

        return {'load_namespace': load_namespace, 'support_code': support_code}

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


rand_code = '''
cdef int _rand_buffer_size = 1024 
cdef double[:] _rand_buf = _numpy.zeros(_rand_buffer_size, dtype=_numpy.float64)
cdef int _cur_rand_buf = 0
cdef double rand(int _idx):
    global _cur_rand_buf
    global _rand_buf
    if _cur_rand_buf==0:
        _rand_buf = _numpy.random.rand(_rand_buffer_size)
    cdef double val = _rand_buf[_cur_rand_buf]
    _cur_rand_buf = (_cur_rand_buf+1)%_rand_buffer_size
    return val
'''

randn_code = rand_code.replace('rand', 'randn').replace('randnom', 'random')

DEFAULT_FUNCTIONS['rand'].implementations.add_implementation(CythonCodeGenerator,
                                                             code=rand_code,
                                                             name='rand')

DEFAULT_FUNCTIONS['randn'].implementations.add_implementation(CythonCodeGenerator,
                                                              code=randn_code,
                                                              name='randn')

int_code = '''
ctypedef fused _to_int:
    char
    short
    int
    float
    double

cdef int _int(_to_int x):
    return <int>x
'''
DEFAULT_FUNCTIONS['int'].implementations.add_implementation(CythonCodeGenerator,
                                                            code=int_code,
                                                            name='_int')

clip_code = '''
ctypedef fused _float_or_double:
    float
    double

cdef _float_or_double clip(_float_or_double x, _float_or_double low,
                           _float_or_double high):
    if x<low:
        return low
    if x>high:
        return high
    return x
'''
DEFAULT_FUNCTIONS['clip'].implementations.add_implementation(CythonCodeGenerator,
                                                             code=clip_code,
                                                             name='clip')

