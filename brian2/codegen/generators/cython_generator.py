import itertools

from brian2.devices.device import all_devices
from brian2.utils.stringtools import word_substitute, deindent, indent
from brian2.parsing.rendering import NodeRenderer
from brian2.parsing.bast import brian_dtype_from_dtype
from brian2.core.functions import DEFAULT_FUNCTIONS, Function
from brian2.core.variables import (Constant, AuxiliaryVariable,
                                   get_dtype_str, Variable, Subexpression)

from .base import CodeGenerator


__all__ = ['CythonCodeGenerator']


data_type_conversion_table = [
    # canonical         C++            Numpy
    ('float32',        'float',       'float32'),
    ('float64',        'double',      'float64'),
    ('int32',          'int32_t',     'int32'),
    ('int64',          'int64_t',     'int64'),
    ('bool',           'bool',        'bool'),
    ('uint8',          'char',        'uint8'),
    ('uint64',         'uint64_t',    'uint64'),
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

    def render_BinOp(self, node):
        if node.op.__class__.__name__=='Mod':
            return '((({left})%({right}))+({right}))%({right})'.format(left=self.render_node(node.left),
                                                                       right=self.render_node(node.right))
        else:
            return super(CythonNodeRenderer, self).render_BinOp(node)


class CythonCodeGenerator(CodeGenerator):
    '''
    Cython code generator
    '''

    class_name = 'cython'

    def __init__(self, *args, **kwds):
        self.temporary_vars = set()
        super(CythonCodeGenerator, self).__init__(*args, **kwds)

    def translate_expression(self, expr):
        expr = word_substitute(expr, self.func_name_replacements)
        return CythonNodeRenderer().render_expr(expr, self.variables).strip()

    def translate_statement(self, statement):
        var, op, expr, comment = (statement.var, statement.op,
                                  statement.expr, statement.comment)
        if op == ':=': # make no distinction in Cython (declaration are done elsewhere)
            op = '='
        # For Cython we replace complex expressions involving boolean variables into a sequence of
        # if/then expressions with simpler expressions. This is provided by the optimise_statements
        # function.
        if (statement.used_boolean_variables is not None and len(statement.used_boolean_variables)
                # todo: improve dtype analysis so that this isn't necessary
                and brian_dtype_from_dtype(statement.dtype)=='float'):
            used_boolvars = statement.used_boolean_variables
            bool_simp = statement.boolean_simplified_expressions
            codelines = []
            firstline = True
            # bool assigns is a sequence of (var, value) pairs giving the conditions under
            # which the simplified expression simp_expr holds
            for bool_assigns, simp_expr in bool_simp.iteritems():
                # generate a boolean expression like ``var1 and var2 and not var3``
                atomics = []
                for boolvar, boolval in bool_assigns:
                    if boolval:
                        atomics.append(boolvar)
                    else:
                        atomics.append('not '+boolvar)
                # use if/else/elif correctly
                if firstline:
                    line = 'if '+(' and '.join(atomics))+':'
                else:
                    if len(used_boolvars)>1:
                        line = 'elif '+(' and '.join(atomics))+':'
                    else:
                        line = 'else:'
                line += '\n    '
                line += var + ' ' + op + ' ' + self.translate_expression(simp_expr)
                codelines.append(line)
                firstline = False
            code = '\n'.join(codelines)
        else:
            code = var + ' ' + op + ' ' + self.translate_expression(expr)
        if len(comment):
            code += ' # ' + comment
        return code

    def translate_one_statement_sequence(self, statements, scalar=False):
        # Note that we do not call this function from
        # `translate_statement_sequence` (which has been overwritten)
        # It is nevertheless implemented, so that it can be called explicitly
        # (e.g. from the GSL code generation)
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # index and read arrays (index arrays first)
        lines += self.translate_to_read_arrays(read, indices)
        # the actual code
        lines += self.translate_to_statements(statements, conditional_write_vars)
        # write arrays
        lines += self.translate_to_write_arrays(write)

        return lines

    def translate_to_read_arrays(self, read, indices):
        lines = []
        for varname in itertools.chain(indices, read):
            var = self.variables[varname]
            index = self.variable_indices[varname]
            line = ('{varname} = '
                    '{arrayname}[{index}]').format(varname=varname,
                                                   arrayname=self.get_array_name(
                                                       var),
                                                   index=index)
            lines.append(line)
        return lines

    def translate_to_statements(self, statements, conditional_write_vars):
        lines = []
        for stmt in statements:
            if stmt.op == ':=' and not stmt.var in self.variables:
                self.temporary_vars.add((stmt.var, stmt.dtype))
            line = self.translate_statement(stmt)
            if stmt.var in conditional_write_vars:
                subs = {}
                condvar = conditional_write_vars[stmt.var]
                lines.append('if %s:' % condvar)
                lines.append(indent(line))
            else:
                lines.append(line)
        return lines

    def translate_to_write_arrays(self, write):
        lines = []
        for varname in write:
            index_var = self.variable_indices[varname]
            var = self.variables[varname]
            line = self.get_array_name(var,
                                       self.variables) + '[' + index_var + '] = ' + varname
            lines.append(line)
        return lines

    def translate_statement_sequence(self, sc_statements, ve_statements):
        # This function is overwritten, since we do not want to completely
        # separate the code generation for scalar and vector code

        assert set(sc_statements.keys()) == set(ve_statements.keys())

        sc_code = {}
        ve_code = {}

        for block_name in sc_statements:
            sc_block = sc_statements[block_name]
            ve_block = ve_statements[block_name]
            (sc_read, sc_write,
             sc_indices, sc_cond_write) = self.arrays_helper(sc_block)
            (ve_read, ve_write,
             ve_indices, ve_cond_write) = self.arrays_helper(ve_block)
            # We want to read all scalar variables that are needed in the
            # vector code already in the scalar code, if they are not written
            for varname in set(ve_read):
                var = self.variables[varname]
                if var.scalar and varname not in ve_write:
                    sc_read.add(varname)
                    ve_read.remove(varname)

            for (code, stmts, read, write, indices,
                 cond_write) in [(sc_code, sc_block, sc_read, sc_write,
                                  sc_indices, sc_cond_write),
                                 (ve_code, ve_block, ve_read, ve_write,
                                  ve_indices, ve_cond_write)]:
                lines = []
                # index and read arrays (index arrays first)
                lines += self.translate_to_read_arrays(read, indices)
                # the actual code
                lines += self.translate_to_statements(stmts, cond_write)
                # write arrays
                lines += self.translate_to_write_arrays(write)
                code[block_name] = '\n'.join(lines)

        kwds = self.determine_keywords()
        return sc_code, ve_code, kwds


    def _add_user_function(self, varname, var):
        user_functions = []
        load_namespace = []
        support_code = []
        impl = var.implementations[self.codeobj_class]
        func_code= impl.get_code(self.owner)
        # Implementation can be None if the function is already
        # available in Cython (possibly under a different name)
        if func_code is not None:
            if isinstance(func_code, basestring):
                # Function is provided as Cython code
                # To make namespace variables available to functions, we
                # create global variables and assign to them in the main
                # code
                user_functions.append((varname, var))
                func_namespace = impl.get_namespace(self.owner) or {}
                for ns_key, ns_value in func_namespace.iteritems():
                    load_namespace.append(
                        '# namespace for function %s' % varname)
                    if hasattr(ns_value, 'dtype'):
                        if ns_value.shape == ():
                            raise NotImplementedError((
                            'Directly replace scalar values in the function '
                            'instead of providing them via the namespace'))
                        newlines = [
                            "global _namespace{var_name}",
                            "global _namespace_num{var_name}",
                            "cdef _numpy.ndarray[{cpp_dtype}, ndim=1, mode='c'] _buf_{var_name} = _namespace['{var_name}']",
                            "_namespace{var_name} = <{cpp_dtype} *> _buf_{var_name}.data",
                            "_namespace_num{var_name} = len(_namespace['{var_name}'])"
                        ]
                        support_code.append(
                            "cdef {cpp_dtype} *_namespace{var_name}".format(
                                cpp_dtype=get_cpp_dtype(ns_value.dtype),
                                var_name=ns_key))

                    else:  # e.g. a function
                        newlines = [
                            "_namespace{var_name} = namespace['{var_name}']"
                        ]
                    for line in newlines:
                        load_namespace.append(
                            line.format(cpp_dtype=get_cpp_dtype(ns_value.dtype),
                                        numpy_dtype=get_numpy_dtype(
                                            ns_value.dtype),
                                        var_name=ns_key))
                support_code.append(deindent(func_code))
            elif callable(func_code):
                self.variables[varname] = func_code
                line = '{0} = _namespace["{1}"]'.format(varname, varname)
                load_namespace.append(line)
            else:
                raise TypeError(('Provided function implementation '
                                 'for function %s is neither a string '
                                 'nor callable (is type %s instead)') % (
                                varname,
                                type(func_code)))

        dep_support_code = []
        dep_load_namespace = []
        dep_user_functions = []
        if impl.dependencies is not None:
            for dep_name, dep in impl.dependencies.iteritems():
                if dep_name not in self.variables:
                    self.variables[dep_name] = dep
                    sc, ln, uf = self._add_user_function(dep_name, dep)
                    dep_support_code.extend(sc)
                    dep_load_namespace.extend(ln)
                    dep_user_functions.extend(uf)

        return (support_code + dep_support_code,
                dep_load_namespace + load_namespace,
                dep_user_functions + user_functions)

    def determine_keywords(self):
        from brian2.devices.device import get_device
        device = get_device()
        # load variables from namespace
        load_namespace = []
        support_code = []
        handled_pointers = set()
        user_functions = []
        for varname, var in sorted(self.variables.items()):
            if isinstance(var, Variable) and not isinstance(var, (Subexpression, AuxiliaryVariable)):
                load_namespace.append('_var_{0} = _namespace["_var_{1}"]'.format(varname, varname))
            if isinstance(var, AuxiliaryVariable):
                line = "cdef {dtype} {varname}".format(
                                dtype=get_cpp_dtype(var.dtype),
                                varname=varname)
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
                    load_namespace.append('{0} = _namespace["{1}"]'.format(self.get_array_name(var, False),
                                                                           self.get_array_name(var, False)))

                # This is the "true" array name, not the restricted pointer.
                array_name = device.get_array_name(var)
                pointer_name = self.get_array_name(var)
                if pointer_name in handled_pointers:
                    continue
                if getattr(var, 'ndim', 1) > 1:
                    continue  # multidimensional (dynamic) arrays have to be treated differently
                if get_dtype_str(var.dtype) == 'bool':
                    newlines = ["cdef _numpy.ndarray[char, ndim=1, mode='c', cast=True] _buf_{array_name} = _namespace['{array_name}']",
                                "cdef {cpp_dtype} * {array_name} = <{cpp_dtype} *> _buf_{array_name}.data"]
                else:
                    newlines = ["cdef _numpy.ndarray[{cpp_dtype}, ndim=1, mode='c'] _buf_{array_name} = _namespace['{array_name}']",
                                "cdef {cpp_dtype} * {array_name} = <{cpp_dtype} *> _buf_{array_name}.data"]

                if not var.scalar:
                    newlines += ["cdef int _num{array_name} = len(_namespace['{array_name}'])"]

                if var.scalar and var.constant:
                    newlines += ['cdef {cpp_dtype} {varname} = _namespace["{varname}"]']
                else:
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
                sc, ln, uf = self._add_user_function(varname, var)
                support_code.extend(sc)
                load_namespace.extend(ln)
                user_functions.extend(uf)
            else:
                # fallback to Python object
                load_namespace.append('{0} = _namespace["{1}"]'.format(varname, varname))

        for varname, dtype in sorted(self.temporary_vars):
            cpp_dtype = get_cpp_dtype(dtype)
            line = "cdef {cpp_dtype} {varname}".format(cpp_dtype=cpp_dtype,
                                                       varname=varname)
            load_namespace.append(line)

        return {'load_namespace': '\n'.join(load_namespace),
                'support_code_lines': support_code}

###############################################################################
# Implement functions
################################################################################
# Functions that exist under the same name in C++
for func in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log',
             'log10', 'sqrt', 'ceil', 'floor', 'abs']:
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(CythonCodeGenerator,
                                                               code=None)

# Functions that need a name translation
for func, func_cpp in [('arcsin', 'asin'), ('arccos', 'acos'), ('arctan', 'atan'),
                       ('int', 'int_')  # from stdint_compat.h
                       ]:
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(CythonCodeGenerator,
                                                               code=None,
                                                               name=func_cpp)

_BUFFER_SIZE = 20000

rand_code = '''
cdef double _rand(int _idx):
    cdef double **buffer_pointer = <double**>_namespace_rand_buffer
    cdef double *buffer = buffer_pointer[0]
    cdef _numpy.ndarray _new_rand

    if(_namespace_rand_buffer_index[0] == 0):
        if buffer != NULL:
            free(buffer)
        _new_rand = _numpy.random.rand(_BUFFER_SIZE)
        buffer = <double *>_numpy.PyArray_DATA(_new_rand)
        PyArray_CLEARFLAGS(<_numpy.PyArrayObject*>_new_rand, _numpy.NPY_OWNDATA)
        buffer_pointer[0] = buffer

    cdef double val = buffer[_namespace_rand_buffer_index[0]]
    _namespace_rand_buffer_index[0] += 1
    if _namespace_rand_buffer_index[0] == _BUFFER_SIZE:
        _namespace_rand_buffer_index[0] = 0
    return val
'''.replace('_BUFFER_SIZE', str(_BUFFER_SIZE))

randn_code = rand_code.replace('rand', 'randn').replace('randnom', 'random')

device = all_devices['runtime']
DEFAULT_FUNCTIONS['rand'].implementations.add_implementation(CythonCodeGenerator,
                                                             code=rand_code,
                                                             name='_rand',
                                                             namespace={'_rand_buffer': device.rand_buffer,
                                                                        '_rand_buffer_index': device.rand_buffer_index})

DEFAULT_FUNCTIONS['randn'].implementations.add_implementation(CythonCodeGenerator,
                                                              code=randn_code,
                                                              name='_randn',
                                                              namespace={
                                                                  '_randn_buffer': device.randn_buffer,
                                                                  '_randn_buffer_index': device.randn_buffer_index})

sign_code = '''
ctypedef fused _to_sign:
    char
    short
    int
    float
    double

cdef int _sign(_to_sign x):
    return (0 < x) - (x < 0)
'''
DEFAULT_FUNCTIONS['sign'].implementations.add_implementation(CythonCodeGenerator,
                                                             code=sign_code,
                                                             name='_sign')

clip_code = '''
ctypedef fused _to_clip:
    char
    short
    int
    float
    double

cdef _to_clip _clip(_to_clip x, double low, double high):
    if x < low:
        return <_to_clip?>low
    if x > high:
        return <_to_clip?>high
    return x
'''
DEFAULT_FUNCTIONS['clip'].implementations.add_implementation(CythonCodeGenerator,
                                                             code=clip_code,
                                                             name='_clip')

timestep_code = '''
cdef int _timestep(double t, double dt):
    return <int64_t>((t + 1e-3*dt)/dt)
'''
DEFAULT_FUNCTIONS['timestep'].implementations.add_implementation(CythonCodeGenerator,
                                                                 code=timestep_code,
                                                                 name='_timestep')
