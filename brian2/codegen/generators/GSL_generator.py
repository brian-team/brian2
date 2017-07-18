from brian2.core.variables import AuxiliaryVariable, ArrayVariable, Constant
from brian2.core.functions import Function
from brian2.codegen.translation import make_statements

from brian2.codegen.permutation_analysis import (check_for_order_independence,
                                                 OrderDependenceError)

from brian2.core.preferences import prefs, BrianPreference
from brian2.utils.stringtools import get_identifiers, word_substitute
from brian2.core.functions import DEFAULT_FUNCTIONS
from brian2.parsing.statements import parse_statement
from brian2.codegen.generators import c_data_type
import re

__all__ = ['GSLCodeGenerator', 'GSLWeaveCodeGenerator', 'GSLCythonCodeGenerator']

class IntegrationError(Exception):
    '''
    Error used to signify that GSL was unable to complete integration (in Cython file)
    '''
    pass

class GSLCodeGenerator(object):

    def __init__(self, variables, variable_indices, owner, iterate_all,
                 codeobj_class, name, template_name,
                 override_conditional_write=None,
                 allows_scalar_write=False):

        prefs.codegen.cpp.libraries += ['gsl', 'gslcblas']
        prefs.codegen.cpp.headers += ['<stdio.h>', '<stdlib.h>', '<gsl/gsl_odeiv2.h>', '<gsl/gsl_errno.h>','<gsl/gsl_matrix.h>']
        prefs.codegen.cpp.include_dirs += ['/home/charlee/softwarefolder/gsl-2.3/gsl/']

        self.generator = codeobj_class.original_generator_class(variables, variable_indices, owner, iterate_all,
                                                                codeobj_class, name, template_name,
                                                                override_conditional_write, allows_scalar_write)
        self.method_options = codeobj_class.method_options

    def __getattr__(self, item):
        return getattr(self.generator, item)

    def find_function_names(self):
        variables = self.variables
        names = []
        for var, var_obj in variables.items():
            if isinstance(var_obj, Function):
                names += [var]
        return names

    def is_constant_and_cpp_standalone(self, var_obj):
        '''
        This function returns whether var_obj is a Constant and the device is cpp_standalone
        In this case, Brian replaces the variable with their value in the final cpp-code
        Since with GSL we add the variables to a struct (e.g. p.b = b), and access them later like
        p->b this cpp_standalone feature results in code that looks like p.1.2 = 1.2 or p->1.2
        '''
        # imports here to avoid circular imports
        from brian2.devices.device import get_device
        from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
        device = get_device()
        return isinstance(var_obj, Constant) and isinstance(device, CPPStandaloneDevice)

    # A series of functions that should be overridden by child class:
    def write_dataholder_single(self, var_obj):
        raise NotImplementedError

    def unpack_namespace_single(self, var_obj, in_vector, in_scalar):
        raise NotImplementedError

    # GSL functions that are the same for all target languages:
    def find_differential_variables(self, code):
        diff_vars = {}
        for expr_set in code:
            for expr in expr_set.split('\n'):
                expr = expr.strip(' ')
                try:
                    lhs, op, rhs, comment = parse_statement(expr)
                except ValueError:
                    pass
                m = re.search('_gsl_(.+?)_f([0-9]*)$', lhs)
                if m:
                    diff_vars[m.group(1)] = m.group(2)
        return diff_vars

    def diff_var_to_replace(self, diff_vars):
        variables = self.variables
        to_replace = {}
        for var, diff_num in diff_vars.items():
            lhs = self.var_init('_gsl_{var}_f{ind}'.format(var=var,ind=diff_num), 'const double ')
            to_replace[lhs] = 'f[{ind}]'.format(ind=diff_num)
            var_obj = variables[var]
            array_name = self.generator.get_array_name(var_obj, access_data=True)
            idx_name = '_idx' #TODO: could be dynamic?
            replace_what = '{var} = {array_name}[{idx_name}]'.format(array_name=array_name, idx_name=idx_name, var=var)
            replace_with = '{var} = y[{ind}]'.format(ind=diff_num, var=var)
            to_replace[replace_what] = replace_with
            replace_what = '{array_name}[{idx_name}] = {var}'.format(array_name=array_name,
                                                                     idx_name=idx_name,
                                                                     var=var)
            replace_with = ''
            to_replace[replace_what] = replace_with
        return to_replace

    def get_dimension_code(self, diff_num):
        start_function = self.get_syntax('start_function')
        open_function = self.get_syntax('open_function')
        end_statement = self.get_syntax('end_statement')
        end_function = self.get_syntax('end_function')
        code = '\n{start_function}int set_dimension(size_t * dimension){open_function}'.format(start_function=start_function,
                                                                                            open_function=open_function)
        code += '\n\tdimension[0] = {diff_num}{end_statement}'.format(diff_num=diff_num, end_statement=end_statement)
        code += '\n\treturn GSL_SUCCESS{end_statement}\n'.format(end_statement=end_statement)
        code += end_function
        return code

    def yvector_code(self, diff_vars):
        start_function = self.get_syntax('start_function')
        open_function = self.get_syntax('open_function')
        end_statement = self.get_syntax('end_statement')
        end_function = self.get_syntax('end_function')
        open_cast = self.get_syntax('open_cast')
        close_cast = self.get_syntax('close_cast')
        access_pointer = self.get_syntax('access_pointer')

        allocate_yvector = '\n{start_function}double* assign_memory_y(){open_function}'.format(start_function=start_function,
                                                                                               open_function=open_function)
        allocate_yvector += '\n\treturn {open_c}double *{close_c} malloc({dnum}*sizeof(double))'.format(open_c=open_cast,
                                                                                                        close_c=close_cast,
                                                                                                        dnum=len(diff_vars))
        allocate_yvector += end_statement + end_function

        fill_yvector = ['\n{start_function}int fill_y_vector(dataholder * p, double * y, int _idx)'.format(start_function=start_function)]
        fill_yvector[0] += open_function
        empty_yvector = ['\n{start_function}int empty_y_vector(dataholder * p, double * y, int _idx)'.format(start_function=start_function)]
        empty_yvector[0] += open_function

        for var, diff_num in diff_vars.items():
            array_name = self.generator.get_array_name(self.variables[var], access_data=True)
            fill_yvector += ['\ty[{ind}] = p{access}{var}[_idx]{end_statement}'.format(access=access_pointer,
                                                                                       ind=diff_num,
                                                                                       var=array_name,
                                                                                       end_statement=end_statement)]
            empty_yvector += ['\tp{access}{var}[_idx] = y[{ind}]{end_statement}'.format(access=access_pointer,
                                                                                        ind=diff_num,
                                                                                        var=array_name,
                                                                                        end_statement=end_statement)]
        fill_yvector += ['\treturn GSL_SUCCESS{end_statement}{end_function}'.format(end_statement=end_statement,
                                                                                    end_function=end_function)]
        empty_yvector += ['\treturn GSL_SUCCESS{end_statement}{end_function}'.format(end_statement=end_statement,
                                                                                    end_function=end_function)]

        return allocate_yvector + '\n' + ('\n').join(fill_yvector) + '\n' + ('\n').join(empty_yvector)

    def write_dataholder(self, variables_in_vector):
        end_statement = self.get_syntax('end_statement')
        open_struct = self.get_syntax('open_struct')
        end_struct = self.get_syntax('end_struct')

        code = ['\n'+self.declare('struct', 'dataholder') + open_struct]
        code += ['\n\t'+self.declare('int', '_idx', in_struct=True) + end_statement]
        for var, var_obj in variables_in_vector.items():
            if var == 't' or '_gsl' in var:
                continue
            if self.is_constant_and_cpp_standalone(var_obj):
                continue
            code += ['\t'+self.write_dataholder_single(var_obj)]
        code += [end_struct]
        return ('\n').join(code)

    def find_undefined_variables(self, statements):
        variables = self.variables
        other_variables = {}
        for statement in statements:
            var, op, expr, comment = (statement.var, statement.op,
                                      statement.expr, statement.comment)
            if var not in variables:
                other_variables[var] = AuxiliaryVariable(var, dtype=statement.dtype)
        return other_variables

    def find_used_variables(self, statements, other_variables):
        variables = self.variables
        used_variables = {}
        for statement in statements:
            lhs, op, rhs, comment = (statement.var, statement.op,
                                      statement.expr, statement.comment)
            for var in (get_identifiers(rhs)):
                if var in self.function_names:
                    continue
                try:
                    var_obj = variables[var]
                except KeyError:
                    var_obj = other_variables[var]
                used_variables[var] = var_obj # save as object because this has all needed info (dtype, name, isarray)

        # I don't know a nicer way to do this, the above way misses write variables..
        read, write, _ = self.array_read_write(statements)
        for var in (read|write):
            if var not in used_variables:
                used_variables[var] = variables[var] # will always be array and thus exist in variables

        return used_variables

    def to_replace_vector_vars(self, variables_in_vector, ignore=[]):
        access_pointer = self.get_syntax('access_pointer')
        to_replace = {}
        t_in_code = None
        for var, var_obj in variables_in_vector.items():
            if var_obj.name == 't':
                t_in_code = var_obj
                continue
            if '_gsl' in var or var in ignore:
                continue
            if self.is_constant_and_cpp_standalone(var_obj):
                continue
            if isinstance(var_obj, ArrayVariable):
                pointer_name = self.get_pointer_name(var_obj)
                to_replace[pointer_name] = 'p' + access_pointer + pointer_name
            else:
                to_replace[var] = 'p' + access_pointer + var

        # also make sure t declaration is replaced if in code
        if t_in_code is not None:
            t_declare = self.var_init('t', 'const double ')
            array_name = self.get_pointer_name(t_in_code)
            end_statement = self.get_syntax('end_statement')
            replace_what = '{t_declare} = {array_name}[0]{end_statement}'.format(t_declare=t_declare,
                                                                                 array_name=array_name,
                                                                                 end_statement=end_statement)
            to_replace[replace_what] = ''

        return to_replace

    def unpack_namespace(self, variables_in_vector, variables_in_scalar, ignore=[]):
        code = []
        for var, var_obj in self.variables.items():
            if var in ignore:
                continue
            if self.is_constant_and_cpp_standalone(var_obj):
                continue
            in_vector = var in variables_in_vector
            in_scalar = var in variables_in_scalar
            code += [self.unpack_namespace_single(var_obj, in_vector, in_scalar)]
        return ('\n').join(code)


    def translate_vector_code(self, code_lines, to_replace):
        code = []
        for expr_set in code_lines:
            for line in expr_set.split('\n'): # every line seperate to make tabbing correct
                code += ['\t' + line]
        code = ('\n').join(code)
        code = word_substitute(code, to_replace)

        # special substitute because of limitations of regex word boundaries with variable[_idx]
        for from_sub, to_sub in to_replace.items():
            m = re.search('\[(\w+)\];?$', from_sub)
            if m:
                code = re.sub(re.sub('\[','\[', from_sub), to_sub, code)
        return code

    def translate_scalar_code(self, code_lines, variables_in_scalar, variables_in_vector):
        # translate scalar code
        code = []
        for line in code_lines:
            try:
                var, op, expr, comment = parse_statement(line)
            except ValueError:
                code += [line]
                continue
            m = re.search('([a-z|A-Z|0-9|_]+)$', var)
            actual_var = m.group(1)
            if actual_var in variables_in_scalar.keys():
                code += [line]
            if actual_var in variables_in_vector.keys():
                if var == 't':
                    continue
                code += ['p.{var} {op} {expr} {comment}'.format(
                        var=actual_var, op=op, expr=expr, comment=comment)]
        return ('\n').join(code)

    def translate(self, code, dtype): # TODO: it's not so nice we have to copy the contents of this function..
        '''
        Translates an abstract code block into the target language.
        '''
        scalar_statements = {}
        vector_statements = {}
        for ac_name, ac_code in code.iteritems():
            statements = make_statements(ac_code,
                                         self.variables,
                                         dtype,
                                         optimise=True,
                                         blockname=ac_name)
            scalar_statements[ac_name], vector_statements[ac_name] = statements
        for vs in vector_statements.itervalues():
            # Check that the statements are meaningful independent on the order of
            # execution (e.g. for synapses)
            try:
                if self.has_repeated_indices(vs):  # only do order dependence if there are repeated indices
                    check_for_order_independence(vs,
                                                 self.generator.variables,
                                                 self.generator.variable_indices)
            except OrderDependenceError:
                # If the abstract code is only one line, display it in full
                if len(vs) <= 1:
                    error_msg = 'Abstract code: "%s"\n' % vs[0]
                else:
                    error_msg = ('%d lines of abstract code, first line is: '
                                 '"%s"\n') % (len(vs), vs[0])

        # save function names because self.generator.translate_statement_sequence deletes these from self.variables
        # but we need to know which identifiers we can safely ignore (i.e. we can ignore the functions because they are
        # handled by the original generator)
        self.function_names = self.find_function_names()

        scalar_code, vector_code, kwds = self.generator.translate_statement_sequence(scalar_statements,
                                                 vector_statements)

        ############ translate code for GSL

        # differential variable specific operations
        diff_vars = self.find_differential_variables(code.values())
        to_replace = self.diff_var_to_replace(diff_vars)
        GSL_support_code = self.get_dimension_code(len(diff_vars))
        GSL_support_code += self.yvector_code(diff_vars)

        # analyze all needed variables; if not in self.variables: put in separate dic.
        # also keep track of variables needed for scalar statements and vector statements
        other_variables = self.find_undefined_variables(scalar_statements[None]+vector_statements[None])
        variables_in_scalar = self.find_used_variables(scalar_statements[None], other_variables)
        variables_in_vector = self.find_used_variables(vector_statements[None], other_variables)

        # add code for dataholder struct
        GSL_support_code = self.write_dataholder(variables_in_vector) + GSL_support_code
        # add e.g. _lio_1 --> p._lio_1 to replacer
        to_replace.update(self.to_replace_vector_vars(variables_in_vector,ignore=diff_vars.keys()))
        # write statements that unpack (python) namespace to dataholder struct or local namespace
        GSL_main_code = self.unpack_namespace(variables_in_vector, variables_in_scalar, ['t'])

        # rewrite actual calculations described by vector_code and put them in func
        GSL_support_code += self.func_begin + self.translate_vector_code(vector_code[None],
                                                                         to_replace) + '\n\t' + self.func_end
        # rewrite scalar code, keep variables that are needed in scalar code normal
        # and add variables to dataholder for vector_code
        GSL_main_code += '\n' + self.translate_scalar_code(scalar_code[None],
                                                    variables_in_scalar,
                                                    variables_in_vector)

        scalar_code['GSL'] = GSL_main_code
        #kwds['GSL_settings'] = prefs.GSL.settings
        kwds['GSL_settings'] = self.method_options
        kwds['support_code_lines'] += GSL_support_code.split('\n')
        return scalar_code, vector_code, kwds

class GSLCythonCodeGenerator(GSLCodeGenerator):

    func_begin = '\ncdef int func(double t, const double y[], double f[], void * params):' +\
                  '\n\tcdef dataholder * p = <dataholder *> params' +\
                  '\n\tcdef int _idx = p._idx\n'
    func_end = 'return GSL_SUCCESS'

    def c_data_type(self, dtype):
        return c_data_type(dtype)

    def get_syntax(self, type):
        syntax = {'end_statement' : '',
                  'access_pointer' : '.',
                  'start_function' : 'cdef ',
                  'open_function' : ':',
                  'open_struct' : ':',
                  'end_function' : '',
                  'end_struct' : '',
                  'open_cast' : '<',
                  'close_cast' : '>',
                  'diff_var_declaration' : ''}
        return syntax[type]

    def var_init(self, var, type):
        return var

    def declare(self, type, name, in_struct=False):
        if in_struct:
            return type + ' ' + name
        else:
            return 'cdef ' + type + ' ' + name

    def get_array_name(self, var_obj):
        return self.generator.get_array_name(var_obj)

    def get_pointer_name(self, var_obj):
        return self.get_array_name(var_obj)

    def write_dataholder_single(self, var_obj):
        dtype = self.c_data_type(var_obj.dtype)
        if isinstance(var_obj, ArrayVariable):
            array_name = self.get_array_name(var_obj)
            return '{dtype}* {var}'.format(dtype=dtype, var=array_name)
        else:
            return '{dtype} {var}'.format(dtype=dtype, var=var_obj.name)

    def unpack_namespace_single(self, var_obj, in_vector, in_scalar):
        code = []
        if isinstance(var_obj, ArrayVariable):
            array_name = self.generator.get_array_name(var_obj)
            dtype = self.c_data_type(var_obj.dtype)
            if in_vector:
                code += ['p.{array} = <{dtype} *> _buf_{array}.data'.format(array=array_name, dtype=dtype)]
            if in_scalar:
                code += ['{array} = <{dtype} *> _buf_{array}.data'.format(array=array_name, dtype=dtype)]
        else:
            if in_vector:
                code += ['p.{var} = _namespace["{var}"]'.format(var=var_obj.name)]
            if in_scalar:
                code += ['{var} = _namespace["{var}"]'.format(var=var_obj.name)]
        return ('\n').join(code)

class GSLWeaveCodeGenerator(GSLCodeGenerator):

    func_begin = '\nint func(double t, const double y[], double f[], void * params)\n{' +\
                  '\n\tdataholder * p = (dataholder *) params;' +\
                  '\n\tint _idx = p->_idx;\n'
    func_end = 'return GSL_SUCCESS;\n}'

    def get_syntax(self, type):
        syntax = {'end_statement' : ';',
                  'access_pointer' : '->',
                  'start_function' : '',
                  'open_function' : '\n{',
                  'open_struct' : '\n{',
                  'end_function' : '\n}',
                  'end_struct' : '\n};',
                  'open_cast' : '(',
                  'close_cast' : ')',
                  'diff_var_declaration' : 'const double '}
        return syntax[type]

    def var_init(self, var, type):
        return type + var

    def declare(self, type, name, in_struct=False):
        return '%s %s'%(type, name)

    def get_array_name(self, var_obj):
        return self.generator.get_array_name(var_obj, access_data=False)

    def get_pointer_name(self, var_obj):
        return self.generator.get_array_name(var_obj, access_data=True)

    def write_dataholder_single(self, var_obj):
        dtype = self.generator.c_data_type(var_obj.dtype)
        if isinstance(var_obj, ArrayVariable):
            pointer_name = self.generator.get_array_name(var_obj, access_data=True)
            restrict = self.generator.restrict
            if var_obj.scalar:
                restrict = ''
            return '{dtype}* {res} {var};'.format(dtype=dtype, res=restrict, var=pointer_name)
        else:
            return '{dtype} {var};'.format(dtype=dtype, var=var_obj.name)

    def unpack_namespace_single(self, var_obj, in_vector, in_scalar):
        if isinstance(var_obj, ArrayVariable):
            pointer_name = self.get_pointer_name(var_obj)
            array_name = self.get_array_name(var_obj)
            if in_vector:
                return 'p.{ptr} = {array};'.format(ptr=pointer_name, array=array_name)
            else:
                return ''
        else:
            if in_vector:
                return 'p.{var} = {var};'.format(var=var_obj.name)
            else:
                return ''
