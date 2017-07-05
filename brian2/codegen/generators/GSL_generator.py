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
        to_replace = {}
        diff_vars = self.find_differential_variables(code.values())
        for var, diff_num in diff_vars.items():
            array_name = self.generator.get_array_name(self.variables[var], access_data=True)
            to_replace[self.gsl_var_declaration(var, diff_num)] = 'f[{ind}]'.format(ind=diff_num)
            self.func_fill_yvector += ['\ty[{ind}] = p{ptr}{var}[_idx]{end}'.format(ind=diff_num,
                                                                                     ptr=self.ptrstr,
                                                                                     var=array_name,
                                                                                     end=self.endstr)]
            self.func_empty_yvector += ['\tp{ptr}{var}[_idx] = y[{ind}]{end}'.format(ind=diff_num,
                                                                                      ptr=self.ptrstr,
                                                                                      var=array_name,
                                                                                      end=self.endstr)]

        # make sure we have all the info on used variables
        # TODO: might be a more elegant way to do this or comebine this and above
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
                            value = self.variables[identifier]
                        except KeyError:
                            value = self.other_variables[identifier]
                        if isinstance(value, Function):
                            continue
                        if isinstance(value, ArrayVariable):
                            variable_mapping[identifier] = {}
                            variable_mapping[identifier]['actual'] = self.generator.get_array_name(value, access_data=False)
                            variable_mapping[identifier]['pointer'] = self.generator.get_array_name(value)
                            variable_mapping[identifier]['restrict'] = ''
                            if not prefs.codegen.target == 'cython' and not value.scalar:
                                variable_mapping[identifier]['restrict'] += self.generator.restrict
                        if not identifier==var and (identifier in self.variables or identifier in self.other_variables):
                            if is_vector:
                                if not identifier == 't':
                                    variables_in_vector.add(identifier)
                            else:
                                variables_in_scalar.add(identifier)

        # write code on variables
        for var in variables_in_vector:
            if var in defined_vars or '_gsl' in var:
                continue
            try:
                value = self.variables[var]
            except KeyError:
                value = self.other_variables[var]
            self.struct_dataholder += ['\t'+self.write_dataholder(var)]
            to_replace = self.get_replacer(var, to_replace)
        inits = []
        for var in self.variables:
            in_vector = var in variables_in_vector
            in_scalar = var in variables_in_scalar
            inits += [self.unpack_namespace(var, in_vector, in_scalar)]
        scalar_code['GSL'] = ('\n').join(inits) + '\n'

        # rewrite actual calculations described by vector_code
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

        set_dimension = [self.get_set_dimension.format(num_diff_vars=len(diff_vars))]
        allocate_y_vector = [self.get_allocate_y_vector.format(num_diff_vars=len(diff_vars))]

        self.struct_dataholder += [self.structend]
        self.func_fill_yvector += ['\t'+self.funcend]
        self.func_empty_yvector += ['\t'+self.funcend]
        self.func_end += ['\t'+self.funcend]

        vector_code['GSL'] = ('\n').join(self.struct_dataholder + set_dimension + allocate_y_vector +\
                self.func_fill_yvector + self.func_empty_yvector + self.func_begin + self.func_end)

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
            if actual_var in variables_in_scalar:
                code += [line]
            if actual_var in variables_in_vector:
                if var == 't':
                    continue
                code += ['p.{var} {op} {expr} {comment}'.format(
                        var=actual_var, op=op, expr=expr, comment=comment)]
        scalar_code['GSL'] += ('\n').join(code)

        kwds['GSL_settings'] = prefs.GSL.settings
        return scalar_code, vector_code, kwds

class GSLCythonCodeGenerator(GSLCodeGenerator):

    #TODO: I don't know if this is the right place to save this, in case we will use the generator for more than one codeobject..
    struct_dataholder = ['\ncdef struct dataholder:',
                         '\tint _idx']
    func_fill_yvector = ['\ncdef int fill_y_vector(dataholder * p, double * y, int _idx):']
    func_empty_yvector = ['\ncdef int empty_y_vector(dataholder * p, double * y, int _idx):']
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
    get_allocate_y_vector = '\ncdef double* assign_memory_y():'
    get_allocate_y_vector += '\n\treturn <double *>malloc({num_diff_vars}*sizeof(double))'

    def c_data_type(self, dtype):
        return c_data_type(dtype)

    def gsl_var_declaration(self, var, ind):
        return '_gsl_{var}_f{ind}'.format(var=var, ind=ind)

    def write_dataholder(self, var):
        try:
            var_obj = self.variables[var]
        except:
            var_obj = self.other_variables[var]
        dtype = self.c_data_type(var_obj.dtype)
        if isinstance(var_obj, ArrayVariable):
            array_name = self.generator.get_array_name(var_obj)
            return '{dtype}* {var}'.format(dtype=dtype, var=array_name)
        else:
            return '{dtype} {var}'.format(dtype=dtype, var=var)

    def get_replacer(self, var, to_replace):
        try:
            var_obj = self.variables[var]
        except:
            var_obj = self.other_variables[var]
        if isinstance(var_obj, ArrayVariable):
            pointer_name = self.generator.get_array_name(var_obj)
            to_replace[pointer_name] = 'p.' + pointer_name
        else:
            to_replace[var] = 'p.' + var
        return to_replace

    def unpack_namespace(self, var, in_vector, in_scalar):
        code = []
        try:
            var_obj = self.variables[var]
        except:
            var_obj = self.other_variables[var]
        if isinstance(var_obj, ArrayVariable):
            array_name = self.generator.get_array_name(var_obj)
            dtype = self.c_data_type(var_obj.dtype)
            if in_vector:
                code += ['p.{array} = <{dtype} *> _buf_{array}.data'.format(array=array_name, dtype=dtype)]
            if in_scalar:
                code += ['{array} = <{dtype} *> _buf_{array}.data'.format(array=array_name, dtype=dtype)]
        else:
            if in_vector:
                code += ['p.{var} = _namespace["{var}"]'.format(var=var)]
            if in_scalar:
                code += ['{var} = _namespace["{var}"]'.format(var=var)]
        return ('\n').join(code)

class GSLWeaveCodeGenerator(GSLCodeGenerator):

    struct_dataholder = ['\nstruct dataholder\n{',
                         '\tint _idx;']
    func_fill_yvector = ['\nint fill_y_vector(dataholder * p, double * y, int _idx)\n{']
    func_empty_yvector = ['\nint empty_y_vector(dataholder * p, double * y, int _idx)\n{']
    func_begin = ['int func(double t, const double y[], double f[], void * params)\n{',
                  '\tdataholder * p = (dataholder *) params;',
                  '\tint _idx = p->_idx;']
    func_end = []

    ptrstr = '->'
    endstr = ';'
    funcend = 'return GSL_SUCCESS;\n}'
    structend = '\n};'

    get_set_dimension = '\nint set_dimension(size_t * dimension)\n{{'
    get_set_dimension += '\n\tdimension[0] = {num_diff_vars};'
    get_set_dimension += '\n\treturn GSL_SUCCESS;\n}}'
    get_allocate_y_vector = '\ndouble* assign_memory_y()\n{{'
    get_allocate_y_vector += '\n\treturn (double *)malloc({num_diff_vars}*sizeof(double));\n}}'

    def gsl_var_declaration(self, var, ind):
        #TODO: I don't think const double should be hardcoded, but I think since this will always apply to differnetial variables it will mostly be the case..
        return 'const double _gsl_{var}_f{ind}'.format(var=var, ind=ind)

    def write_dataholder(self, var):
        try:
            var_obj = self.variables[var]
        except:
            var_obj = self.other_variables[var]
        dtype = self.generator.c_data_type(var_obj.dtype)
        if isinstance(var_obj, ArrayVariable):
            pointer_name = self.generator.get_array_name(var_obj, access_data=True)
            restrict = self.generator.restrict
            if var_obj.scalar:
                restrict = ''
            return '{dtype}* {res} {var};'.format(dtype=dtype, res=restrict, var=pointer_name)
        else:
            return '{dtype} {var};'.format(dtype=dtype, var=var)

    def get_replacer(self, var, to_replace):
        try:
            var_obj = self.variables[var]
        except:
            var_obj = self.other_variables[var]
        if isinstance(var_obj, ArrayVariable):
            pointer_name = self.generator.get_array_name(var_obj, access_data=True)
            to_replace[pointer_name] = 'p->' + pointer_name
        else:
            to_replace[var] = 'p->' + var
        return to_replace

    def unpack_namespace(self, var, in_vector, in_scalar):
        try:
            var_obj = self.variables[var]
        except:
            var_obj = self.other_variables[var]
        if isinstance(var_obj, ArrayVariable):
            pointer_name = self.generator.get_array_name(var_obj, access_data=True)
            array_name = self.generator.get_array_name(var_obj, access_data=False)
            if in_vector:
                return 'p.{ptr} = {array};'.format(ptr=pointer_name, array=array_name)
            else:
                return ''
        else:
            if in_vector:
                return 'p.{var} = {var};'.format(var=var)
            else:
                return ''

