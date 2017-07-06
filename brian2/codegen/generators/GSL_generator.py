from brian2.core.variables import AuxiliaryVariable, ArrayVariable
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

def is_dictionary(obj):
    '''
    Used as validator for GSL settings in BrianGlobalPreferences
    :param obj: object to test
    :return: True if obj is dictionary
    '''
    return isinstance(obj, dict)

#TODO: Change documentation of register_prefernce, the preference keywords should be given without quotation marks!!
# register GSL.settings preference that can be set to change GSL behavior. This dictionary
# is sent to the templater as keyword under GSL_settings
prefs.register_preferences(
    'GSL',
    'Code generation preferences for the C language',
    settings = BrianPreference(
        validator=is_dictionary,
        docs='...',
        default={
            'integrator' : 'rkf45',
            'adaptable_timestep' : True,
            'h_start' : 1e-5,
            'eps_abs' : 1e-6,
            'eps_rel' : 0.
        }))

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

    def __getattr__(self, item):
        return getattr(self.generator, item)

    # A series of functions that should be overridden by child class:
    def diff_var_to_replace(self, diff_vars):
        raise NotImplementedError
    def get_dimension_code(self, diff_num):
        raise NotImplementedError
    def yvector_code(self, diff_vars):
        raise NotImplementedError
    def write_dataholder_single(self, var_obj):
        raise NotImplementedError
    def write_dataholder(self, variables_in_vector):
        raise NotImplementedError
    def to_replace_vector_vars(self, variables_in_vector, ignore=[]):
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

    def find_undefined_variables(self, statements):
        variables = self.variables
        other_variables = {}
        for statement in statements:
            var, op, expr, comment = (statement.var, statement.op,
                                      statement.expr, statement.comment)
            if var not in variables:
                other_variables[var] = AuxiliaryVariable(var, dtype=statement.dtype)
        return other_variables

    def find_used_variables(self, statements, other_variables, ignore=[]):
        variables = self.variables
        used_variables = set()
        for statement in statements:
            lhs, op, rhs, comment = (statement.var, statement.op,
                                      statement.expr, statement.comment)
            for var in (get_identifiers(rhs)):
                if var in DEFAULT_FUNCTIONS or var in ignore: # we don't want functions in the dataholder
                    continue
                try:
                    var_obj = variables[var]
                except KeyError:
                    var_obj = other_variables[var]
                if isinstance(var_obj, Function): # we don't want functions in the dataholder
                    continue
                used_variables.add(var_obj) # save as object because this has all needed info (dtype, name, isarray)
        return used_variables

    def unpack_namespace(self, variables_in_vector, variables_in_scalar):
        code = []
        for var_obj in self.variables.values():
            in_vector = var_obj in variables_in_vector
            in_scalar = var_obj in variables_in_scalar
            code += [self.unpack_namespace_single(var_obj, in_vector, in_scalar)]
        return ('\n').join(code)


    def translate_vector_code(self, code_lines, to_replace, ignore=[]): # TODO: ignore 't'?
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
            if actual_var in [var_obj.name for var_obj in variables_in_scalar]:
                code += [line]
            if actual_var in [var_obj.name for var_obj in variables_in_vector]:
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
        variables_in_vector = self.find_used_variables(vector_statements[None], other_variables, ['t'])

        # add code for dataholder struct
        GSL_support_code = self.write_dataholder(variables_in_vector) + GSL_support_code
        # add e.g. _lio_1 --> p._lio_1 to replacer
        to_replace.update(self.to_replace_vector_vars(variables_in_vector,ignore=['t']+diff_vars.keys()))
        # write statements that unpack (python) namespace to dataholder struct or local namespace
        GSL_main_code = self.unpack_namespace(variables_in_vector, variables_in_scalar)

        # rewrite actual calculations described by vector_code and put them in func
        GSL_support_code += self.func_begin + self.translate_vector_code(vector_code[None],
                                                                         to_replace,
                                                                         ignore=['t']) + '\n\t' + self.func_end
        # rewrite scalar code, keep variables that are needed in scalar code normal
        # and add variables to dataholder for vector_code
        GSL_main_code += self.translate_scalar_code(scalar_code[None],
                                                    variables_in_scalar,
                                                    variables_in_vector)

        scalar_code['GSL'] = GSL_main_code
        vector_code['GSL'] = GSL_support_code
        kwds['GSL_settings'] = prefs.GSL.settings
        return scalar_code, vector_code, kwds

class GSLCythonCodeGenerator(GSLCodeGenerator):

    func_begin = '\ncdef int func(double t, const double y[], double f[], void * params):' +\
                  '\n\tcdef dataholder * p = <dataholder *> params' +\
                  '\n\tcdef int _idx = p._idx\n'
    func_end = 'return GSL_SUCCESS'

    def c_data_type(self, dtype):
        return c_data_type(dtype)

    def get_dimension_code(self, diff_num):
        code = '\ncdef int set_dimension(size_t * dimension):'
        code += '\n\tdimension[0] = %d'%diff_num
        code += '\n\treturn GSL_SUCCESS\n'
        return code

    def diff_var_to_replace(self, diff_vars):
        variables = self.variables
        to_replace = {}
        for var, diff_num in diff_vars.items():
            lhs = 'const double _gsl_{var}_f{ind}'.format(var=var, ind=diff_num)
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

    def diff_var_to_replace(self, diff_vars):
        variables = self.variables
        to_replace = {}
        for var, diff_num in diff_vars.items():
            lhs = '_gsl_{var}_f{ind}'.format(var=var, ind=diff_num)
            to_replace[lhs] = 'f[{ind}]'.format(ind=diff_num)
            var_obj = variables[var]
            array_name = self.generator.get_array_name(var_obj)
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

    def yvector_code(self, diff_vars):
        allocate_yvector = '\ncdef double* assign_memory_y():'
        allocate_yvector += '\n\treturn <double *>malloc(%d*sizeof(double))'%len(diff_vars)
        fill_yvector = ['\ncdef int fill_y_vector(dataholder * p, double * y, int _idx):']
        empty_yvector = ['\ncdef int empty_y_vector(dataholder * p, double * y, int _idx):']
        for var, diff_num in diff_vars.items():
            array_name = self.generator.get_array_name(self.variables[var])
            fill_yvector += ['\ty[{ind}] = p.{var}[_idx]'.format(ind=diff_num,
                                                                   var=array_name)]
            empty_yvector += ['\tp.{var}[_idx] = y[{ind}]'.format(ind=diff_num,
                                                                   var=array_name)]
        fill_yvector += ['\treturn GSL_SUCCESS\n']
        empty_yvector += ['\treturn GSL_SUCCESS\n']
        return allocate_yvector + '\n' + ('\n').join(fill_yvector) + '\n' + ('\n').join(empty_yvector)

    def write_dataholder_single(self, var_obj):
        dtype = self.c_data_type(var_obj.dtype)
        if isinstance(var_obj, ArrayVariable):
            array_name = self.generator.get_array_name(var_obj)
            return '{dtype}* {var}'.format(dtype=dtype, var=array_name)
        else:
            return '{dtype} {var}'.format(dtype=dtype, var=var_obj.name)

    def write_dataholder(self, variables_in_vector):
        code = ['\ncdef struct dataholder:\n\tint _idx']
        for var_obj in variables_in_vector:
            if var_obj.name == 't' or '_gsl' in var_obj.name:
                continue
            code += ['\t'+self.write_dataholder_single(var_obj)]
        return ('\n').join(code)

    def to_replace_vector_vars(self, variables_in_vector, ignore=[]):
        to_replace = {}
        for var_obj in variables_in_vector:
            if var_obj.name in ignore or '_gsl' in var_obj.name:
                continue
            if isinstance(var_obj, ArrayVariable):
                array_name = self.generator.get_array_name(var_obj)
                to_replace[array_name] = 'p.' + array_name
            else:
                var = var_obj.name
                to_replace[var] = 'p.' + var
        return to_replace

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
                  '\tdataholder * p = (dataholder *) params;' +\
                  '\tint _idx = p->_idx;'
    func_end = '\treturn GSL_SUCCESS;\n}'

    def diff_var_to_replace(self, diff_vars):
        variables = self.variables
        to_replace = {}
        for var, diff_num in diff_vars.items():
            lhs = 'const double _gsl_{var}_f{ind}'.format(var=var, ind=diff_num)
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
        return ('\nint set_dimension(size_t * dimension)\n{' +\
               '\n\tdimension[0] = %d;' +\
               '\n\treturn GSL_SUCCESS;\n}')%diff_num

    def yvector_code(self, diff_vars):
        allocate_yvector = '\ndouble* assign_memory_y()\n{'
        allocate_yvector += '\n\treturn (double *)malloc(%d*sizeof(double));\n}'%len(diff_vars)
        fill_yvector = ['\nint fill_y_vector(dataholder * p, double * y, int _idx)\n{']
        empty_yvector = ['\nint empty_y_vector(dataholder * p, double * y, int _idx)\n{']
        for var, diff_num in diff_vars.items():
            array_name = self.generator.get_array_name(self.variables[var], access_data=True)
            fill_yvector += ['\ty[{ind}] = p->{var}[_idx];'.format(ind=diff_num,
                                                                   var=array_name)]
            empty_yvector += ['\tp->{var}[_idx] = y[{ind}];'.format(ind=diff_num,
                                                                   var=array_name)]
        fill_yvector += ['\treturn GSL_SUCCESS;\n}']
        empty_yvector += ['\treturn GSL_SUCCESS;\n}']
        return allocate_yvector + '\n' + ('\n').join(fill_yvector) + '\n' + ('\n').join(empty_yvector)

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

    def write_dataholder(self, variables_in_vector):
        code = ['\nstruct dataholder\n{\n\tint _idx;']
        for var_obj in variables_in_vector:
            if var_obj.name == 't' or '_gsl' in var_obj.name:
                continue
            code += ['\t'+self.write_dataholder_single(var_obj)]
        code += ['\n};']
        return ('\n').join(code)

    def to_replace_vector_vars(self, variables_in_vector, ignore=[]):
        to_replace = {}
        for var_obj in variables_in_vector:
            if var_obj.name in ignore or '_gsl' in var_obj.name:
                continue
            if isinstance(var_obj, ArrayVariable):
                pointer_name = self.generator.get_array_name(var_obj, access_data=True)
                to_replace[pointer_name] = 'p->' + pointer_name
            else:
                var = var_obj.name
                to_replace[var] = 'p->' + var
        return to_replace

    def unpack_namespace_single(self, var_obj, in_vector, in_scalar):
        if isinstance(var_obj, ArrayVariable):
            pointer_name = self.generator.get_array_name(var_obj, access_data=True)
            array_name = self.generator.get_array_name(var_obj, access_data=False)
            if in_vector:
                return 'p.{ptr} = {array};'.format(ptr=pointer_name, array=array_name)
            else:
                return ''
        else:
            if in_vector:
                return 'p.{var} = {var};'.format(var=var_obj.name)
            else:
                return ''
