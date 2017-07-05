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

class GSLCodeGenerator(object): #TODO: I don't think it matters it doesn't inherit from CodeGenerator (the base) because it can access this through __getattr__ of the parent anyway?

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

    def gsl_var_declaration(self, *args, **kwargs):
        raise NotImplementedError

    def write_dataholder(self, *args, **kwargs):
        raise NotImplementedError

    def get_replacer(self, *args, **kwargs):
        raise NotImplementedError

    def unpack_namespace(self, *args, **kwargs):
        raise NotImplementedError

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

        # translate code for GSL
        defined_vars = ['t']

        # differential variable specific operations
        diff_vars = self.find_differential_variables(code.values())
        to_replace = self.diff_var_to_replace(diff_vars)
        GSL_support_code = self.get_dimension_code(len(diff_vars))
        GSL_support_code += self.yvector_code(diff_vars)

        # make sure we have all the info on used variables
        self.other_variables = {}
        variable_mapping = {}
        read = set()
        variables_in_vector = set()
        variables_in_scalar = set()
        for is_vector, dictionary in zip([False, True], [scalar_statements, vector_statements]):
            for key, value in dictionary.items():
                for statement in value:
                    var, op, expr, comment = (statement.var, statement.op,
                                             statement.expr, statement.comment)
                    if var not in self.variables:
                        self.other_variables[var] = AuxiliaryVariable(var, dtype=statement.dtype)
                    for identifier in (set([var])|get_identifiers(expr)):
                        if identifier in DEFAULT_FUNCTIONS: #TODO: also DEFAULT_CONSTANTS?
                            continue
                        read.add(identifier)
                        try:
                            var_obj = self.variables[identifier]
                        except KeyError:
                            var_obj = self.other_variables[identifier]
                        if isinstance(value, Function):
                            continue
                        if not identifier==var and (identifier in self.variables or identifier in self.other_variables):
                            if is_vector:
                                if not identifier == 't':
                                    variables_in_vector.add(var_obj)
                            else:
                                variables_in_scalar.add(var_obj)

        # add code for dataholder struct
        GSL_support_code = self.write_dataholder(variables_in_vector) + GSL_support_code
        # add e.g. _lio_1 --> p._lio_1 to replacer
        to_replace.update(self.to_replace_vector_vars(variables_in_vector))
        # write statements that unpack namespace to dataholder struct or local namespace
        GSL_main_code = self.unpack_namespace(variables_in_vector, variables_in_scalar)

        # rewrite actual calculations described by vector_code and put them in func
        GSL_support_code += self.func_begin
        for expr_set in vector_code[None]:
            for line in expr_set.split('\n'):
                try:
                    var_original, op, expr, comment = parse_statement(line)
                    m = re.search('([a-z|A-Z|0-9|_|\[|\]]+)$', var_original)
                    var = m.group(1)
                except ValueError:
                    self.func_end += ['\t'+line]
                    continue
                if var in diff_vars or expr in diff_vars:
                    pointer_name = self.generator.get_array_name(self.variables[var], access_data=True)
                    if pointer_name in expr: # v = _array_etc[_idx] should be set by y instead
                        self.func_end += ['\t{var} = y[{ind}]{end}'.format(var=var_original,
                                                                           ind=diff_vars[var],
                                                                           end=self.endstr)]
                        continue
                    elif pointer_name in var: # and e.g. _array_neurongroup_v[_idx] = v should be ignored
                        continue
                if var in defined_vars: # ignore t = _array_defaultclock_t[0]
                    continue
                self.func_end += ['\t'+word_substitute(line, to_replace)]

        self.func_end += ['\t'+self.funcend]

        vector_code['GSL'] = GSL_support_code + ('\n').join(self.func_end)

        # translate scalar code
        code = []
        for line in scalar_code[None]:
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
        GSL_main_code += ('\n').join(code)

        scalar_code['GSL'] = GSL_main_code
        kwds['GSL_settings'] = prefs.GSL.settings
        return scalar_code, vector_code, kwds

class GSLCythonCodeGenerator(GSLCodeGenerator):

    #TODO: I don't know if this is the right place to save this, in case we will use the generator for more than one codeobject..
    struct_dataholder = ['\ncdef struct dataholder:',
                         '\tint _idx']
    fill_yvector = ['\ncdef int fill_y_vector(dataholder * p, double * y, int _idx):']
    empty_yvector = ['\ncdef int empty_y_vector(dataholder * p, double * y, int _idx):']
    func_begin = ['cdef int func(double t, const double y[], double f[], void * params):',
                  '\tcdef dataholder * p = <dataholder *> params',
                  '\tcdef int _idx = p._idx']
    func_end = []

    ptrstr = '.'
    endstr = ''
    funcend = 'return GSL_SUCCESS'
    structend = ''

    get_set_dimension = '\ncdef int set_dimension(size_t * dimension):'
    get_set_dimension += '\n\tdimension[0] = {num_diff_vars}'
    get_set_dimension += '\n\treturn GSL_SUCCESS'

    def c_data_type(self, dtype):
        return c_data_type(dtype)

    def diff_var_to_replace(self, diff_vars):
        to_replace = {}
        for var, diff_num in diff_vars.items():
            lhs = '_gsl_{var}_f{ind}'.format(var=var, ind=diff_num)
            to_replace[lhs] = 'f[{ind}]'.format(ind=diff_num)
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
        fill_yvector += ['\treturn GSL_SUCCESS']
        empty_yvector += ['\treturn GSL_SUCCESS']
        return allocate_yvector + '\n' + ('\n').join(fill_yvector) + '\n' + ('\n').join(empty_yvector)

    def write_dataholder(self, var_obj):
        dtype = self.c_data_type(var_obj.dtype)
        if isinstance(var_obj, ArrayVariable):
            array_name = self.generator.get_array_name(var_obj)
            return '{dtype}* {var}'.format(dtype=dtype, var=array_name)
        else:
            return '{dtype} {var}'.format(dtype=dtype, var=var_obj.name)

    def get_replacer(self, var_obj, to_replace):
        if isinstance(var_obj, ArrayVariable):
            pointer_name = self.generator.get_array_name(var_obj)
            to_replace[pointer_name] = 'p.' + pointer_name
        else:
            var = var_obj.name
            to_replace[var] = 'p.' + var
        return to_replace

    def unpack_namespace(self, var_obj, in_vector, in_scalar):
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

    func_begin = ('\n').join(['\nint func(double t, const double y[], double f[], void * params)\n{',
                  '\tdataholder * p = (dataholder *) params;',
                  '\tint _idx = p->_idx;'])

    func_end = []
    endstr = ';'
    funcend = '\treturn GSL_SUCCESS;\n}'

    def diff_var_to_replace(self, diff_vars):
        to_replace = {}
        for var, diff_num in diff_vars.items():
            lhs = 'const double _gsl_{var}_f{ind}'.format(var=var, ind=diff_num)
            to_replace[lhs] = 'f[{ind}]'.format(ind=diff_num)
        return to_replace

    def get_dimension_code(self, diff_num):
        code = '\nint set_dimension(size_t * dimension)\n{'
        code += '\n\tdimension[0] = %d;'%diff_num
        code += '\n\treturn GSL_SUCCESS;\n}'
        return code

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

    def to_replace_vector_vars(self, variables_in_vector):
        to_replace = {}
        for var_obj in variables_in_vector:
            if var_obj.name == 't' or '_gsl' in var_obj.name:
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

    def unpack_namespace(self, variables_in_vector, variables_in_scalar):
        code = []
        for var_obj in self.variables.values():
            in_vector = var_obj in variables_in_vector
            in_scalar = var_obj in variables_in_scalar
            code += [self.unpack_namespace_single(var_obj, in_vector, in_scalar)]
        return ('\n').join(code)
