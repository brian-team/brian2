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

# default method_options
default_method_options = {
    'integrator' : 'rkf45',
    'adaptable_timestep' : True,
    'h_start' : 1e-5,
    'eps_abs' : 1e-6,
    'eps_rel' : 0.
}

class GSLCodeGenerator(object):

    def __init__(self, variables, variable_indices, owner, iterate_all,
                 codeobj_class, name, template_name,
                 override_conditional_write=None,
                 allows_scalar_write=False):

        prefs.codegen.cpp.libraries += ['gsl', 'gslcblas']
        prefs.codegen.cpp.headers += ['<stdio.h>', '<stdlib.h>', '<gsl/gsl_odeiv2.h>', '<gsl/gsl_errno.h>','<gsl/gsl_matrix.h>']
        prefs.codegen.cpp.include_dirs += ['/home/charlee/softwarefolder/gsl-2.3/gsl/'] #TODO: obviously this has to be changed

        self.generator = codeobj_class.original_generator_class(variables, variable_indices, owner, iterate_all,
                                                                codeobj_class, name, template_name,
                                                                override_conditional_write, allows_scalar_write)
        self.method_options = default_method_options
        if not codeobj_class.method_options is None:
            for key, value in codeobj_class.method_options.items():
                self.method_options[key] = value
        self.variable_flags = codeobj_class.variable_flags #TODO: temporary solution for sending flags to generator

    def __getattr__(self, item):
        return getattr(self.generator, item)

    # A series of functions that should be overridden by child class:
    def c_data_type(self, dtype):
        '''
        Get string version of object dtype that is attached to Brian variables. cpp_generator already has this function, but
        the Cython generator does not, but we need it for GSL code generation.
        '''
        return NotImplementedError

    def var_init_lhs(self, var, type):
        '''
        What does the left hand side of an initializing expression look like? (i.e. for cpp this is const double var = ...
        while in cython it is just var = ...
        '''
        raise NotImplementedError

    def get_array_name(self, var_obj):
        '''
        Get the array_name used in Python Brian
        '''
        raise NotImplementedError

    def get_pointer_name(self, var_obj):
        '''
        Get the pointer_name used to refer to array object. Only differs for cpp, but function also defined for cython,
        this just returns the same as get_array_name (for maximum generalizability of the GSL_generator code)
        '''
        raise NotImplementedError

    def unpack_namespace_single(self, var_obj, in_vector, in_scalar):
        '''
        Writes the code necessary to pull single variable out of the Brian namespace into the generated code.
        The code created is significantly different between cpp and cython, so I decided to not make this function general
        over all target languages (i.e. in contrast to most other functions that only have syntactical differences)
        '''
        raise NotImplementedError

    # GSL functions that are the same for all target languages:
    def find_function_names(self):
        '''
        :return: list of function names in the self.variables dictionary
        '''
        variables = self.variables
        names = []
        for var, var_obj in variables.items():
            if isinstance(var_obj, Function):
                names += [var]
        return names

    def is_cpp_standalone(self):
        '''
        Function checks whether we're running with the cpp_standalone device. (see description of is_constant_and_cpp_standalone)
        :returns boolean whether current device is cpp_standalone
        '''
        # imports here to avoid circular imports
        from brian2.devices.device import get_device
        from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
        device = get_device()
        return isinstance(device, CPPStandaloneDevice)

    def is_constant_and_cpp_standalone(self, var_obj):
        '''
        This function returns whether var_obj is a Constant and the device is cpp_standalone
        In this case (Constants with cpp_standalone device), Brian replaces the variable with their value in the final cpp-code
        Since with GSL we add the variables to a struct (e.g. p.b = b), and access them later like
        p->b this cpp_standalone feature results in code that looks like p.1.2 = 1.2 or p->1.2.
        :return: boolean describing whether the variable object that was given is Constant and the current device is cpp_standalone
        '''
        return isinstance(var_obj, Constant) and self.cpp_standalone

    def find_differential_variables(self, code):
        '''
        Find the variables that were tagged _gsl_{var}_f{ind} by the GSL StateUpdateMethod (with regular expressions)
        :return: dictionary with variable name and differential equation index
        '''
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
        '''
        From the code generated by Brian's 'normal' generators (cpp_generator or cython_generator a few bits of text
        need to be replaced to get GSL compatible code. The bits of text related to differential equation variables
        are put in the replacer dictionary in this function.
        Examples for for example the variable 'v':
        _gsl_v_f0 has to be replaced with f[0] (and in the case of cpp the lhs is const double _gsl_v_f0)
        v = _array_neurongroup_v[_idx] has to be replaced with v = y[0]
        :param diff_vars: dictionary with mapping between differential equation variable and their index (given by GSL StateUpdateMethod)
        :return: dictionary with bits of string that should be replaced to change Brian normal code to GSL code
        '''
        variables = self.variables
        to_replace = {}
        for var, diff_num in diff_vars.items():
            to_replace.update(self.var_replace_diff_var_lhs(var, diff_num))
            var_obj = variables[var]
            array_name = self.generator.get_array_name(var_obj, access_data=True)
            idx_name = '_idx' #TODO: could be dynamic?
            replace_what = '{var} = {array_name}[{idx_name}]'.format(array_name=array_name, idx_name=idx_name, var=var)
            replace_with = '{var} = y[{ind}]'.format(ind=diff_num, var=var)
            to_replace[replace_what] = replace_with
        return to_replace

    def get_dimension_code(self, diff_num):
        '''
        GSL needs to know how many differential variables there are in the ODE system. Since the current approach is to have
        the code in the vector loop the same for all simulations, this dimension is set by an external function as well.
        The code for this set_dimension functon is written here. It is assumed the code will be the same for each target
        language with the exception of some syntactical differences
        :param diff_num: number of differential equation variables
        :return: string with code describing the function that sets the dimension of the ODE system
        '''
        code = ['\n{start_declare}int set_dimension(size_t * dimension){open_function}']
        code += ['\tdimension[0] = %d{end_statement}'%diff_num]
        code += ['\treturn GSL_SUCCESS{end_statement}{end_function}']
        return ('\n').join(code).format(**self.syntax)

    def yvector_code(self, diff_vars):
        '''
        The values of differential variables have to be transferred from Brian's namespace to a vector that is given to
        GSL. The allocation of this vector and the transferring from Brian --> y and back from y --> Brian after integration
        has happened happens in separate functions. These are written here.
        :param diff_vars:
        :return: string with code describing three functions related to the GSL y vector
        '''
        allocate_y = ['\n{start_declare}double* assign_memory_y(){open_function}']
        allocate_y += ['\treturn {open_cast}double *{close_cast} malloc(%d*sizeof(double))'%len(diff_vars)]
        allocate_y[-1] += '{end_statement}{end_function}'
        fill_y = ['\n{start_declare}int fill_y_vector(dataholder * p, double * y, int _idx){open_function}']
        empty_y = ['\n{start_declare}int empty_y_vector(dataholder * p, double * y, int _idx){open_function}']
        for var, diff_num in diff_vars.items():
            diff_num = int(diff_num)
            array_name = self.generator.get_array_name(self.variables[var], access_data=True)
            fill_y += ['\ty[%d] = p{access_pointer}%s[_idx]{end_statement}'%(diff_num, array_name)]
            empty_y += ['\tp{access_pointer}%s[_idx] = y[%d]{end_statement}'%(array_name, diff_num)]
        fill_y += ['\treturn GSL_SUCCESS{end_statement}{end_function}']
        empty_y += ['\treturn GSL_SUCCESS{end_statement}{end_function}']
        return ('\n').join(allocate_y + fill_y + empty_y).format(**self.syntax)

    def make_function_code(self, lines):
        '''
        Adds nonchanging aspects of GSL func code to lines of code written somewhere else (translate_vector_code).
        :param lines: string of code decribing the system of equations
        :return: string with the complete function
        '''
        code = ['\n']
        code += ['{start_declare}int func(double t, const double y[], double f[], void * params){open_function}']
        code += ['\t{start_declare}dataholder * p = {open_cast}dataholder *{close_cast} params{end_statement}']
        code += ['\t{start_declare}int _idx = p{access_pointer}_idx{end_statement}']
        code += [lines]
        code += ['\treturn GSL_SUCCESS{end_statement}{end_function}']
        return ('\n').join(code).format(**self.syntax)

    def write_dataholder_single(self, var_obj):
        '''
        :param var_obj: Variable Object
        :return: string describing this variable object as required for the dataholder struct
        '''
        dtype = self.c_data_type(var_obj.dtype)
        if isinstance(var_obj, ArrayVariable):
            pointer_name = self.get_pointer_name(var_obj)
            try:
                restrict = self.generator.restrict
            except AttributeError:
                restrict = ''
            if var_obj.scalar:
                restrict = ''
            return '%s* %s %s{end_statement}'%(dtype, restrict, pointer_name)
        else:
            return '%s %s{end_statement}'%(dtype, var_obj.name)

    def write_dataholder(self, variables_in_vector):
        '''
        :param variables_in_vector: dictionary mapping between variable names and their Brian objects
        :return: code decribing the dataholder struct that will contain all data necessary for GSL func
        '''
        code = ['\n{start_declare}struct dataholder{open_struct}']
        code += ['\tint _idx{end_statement}']
        for var, var_obj in variables_in_vector.items():
            if var == 't' or '_gsl' in var or self.is_constant_and_cpp_standalone(var_obj):
                continue
            code += ['\t'+self.write_dataholder_single(var_obj)]
        code += ['{end_struct}']
        return ('\n').join(code).format(**self.syntax)

    def find_undefined_variables(self, statements):
        '''
        Brian does not save the _lio_ variables it uses anywhere. This is problematic for our GSL implementation because
        we save the lio variables in the dataholder struct. For this reason, we check all left hand side variables that
        occur and add them to a separate dictionary (could potentially also catch other helper variables)
        :param statements: list of statement objects (need to have the dtype attribute)
        :return: dictionary of variables that are not in self.variables, the objects are defined as 'AuxiliaryVariables'
        and have the correct dtype
        '''
        #TODO: not sure if it is necessary to have other_variables separate from self.variables
        variables = self.variables
        other_variables = {}
        for statement in statements:
            var, op, expr, comment = (statement.var, statement.op,
                                      statement.expr, statement.comment)
            if var not in variables:
                other_variables[var] = AuxiliaryVariable(var, dtype=statement.dtype)
        return other_variables

    def find_used_variables(self, statements, other_variables):
        '''
        :param statements:
        :param other_variables: dictionary of variables that are not in self.variables
        :return: dictionary of variables that are used in the right hand side of the statements given
        '''
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
        '''
        :param variables_in_vector:
        :param ignore:
        :return: dictionary with strings that need to be replaced. i.e. _lio_1 will be p._lio_1 (in cython) or p->_lio_1 (cpp)
         in addition t will always be added because GSL defines its own t
        '''
        #TODO: have t work with t's other than defaultclock?
        access_pointer = self.syntax['access_pointer']
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
            t_declare = self.var_init_lhs('t', 'const double ')
            array_name = self.get_pointer_name(t_in_code)
            end_statement = self.syntax['end_statement']
            replace_what = '{t_declare} = {array_name}[0]{end_statement}'.format(t_declare=t_declare,
                                                                                 array_name=array_name,
                                                                                 end_statement=end_statement)
            to_replace[replace_what] = ''

        return to_replace

    def unpack_namespace(self, variables_in_vector, variables_in_scalar, ignore=[]):
        '''
        Writes code that unpacks Brian namespace to cython/cpp namespace
        For vector code this means putting variables in dataholder (i.e. p->var or p.var = ...)
        Note that code is written so a variable could occur both in scalar and vector code
        :param variables_in_vector: dictionary with variables occurring in vector code
        :param variables_in_scalar: dictionary with variables occurring in scalar code
        :param ignore: string variable names describing variables that should be ignored
        :return: string of code fragment
        '''
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
        '''
        Vector code is translated to code that will be added to func, by adding tabbing and replacing variables with dataholder form
        :param code_lines: lines of code describing vector_code
        :param to_replace: dictionary with to be replaced strings (see to_replace_vector_vars and to_replace_diff_vars)
        :return: new code
        '''
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

        if '_gsl' in code:
            print code
            print('Translation failed, _gsl still in code (should only be tag, and should be replaced)')
            #TODO: raise nicer error
            raise Exception

        return code

    def translate_scalar_code(self, code_lines, variables_in_scalar, variables_in_vector):
        '''
        Translates scalar code. If calculated variables are used in the vector_code their value is added to the variable
        in the dataholder.
        :param code_lines: scalar code as a list of strings
        :param variables_in_scalar: dictionary of variables occurring in scalar code
        :param variables_in_vector: dictionary of variables occurring in vector code
        :return: string of code that should occur in the main before the loop
        '''
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
        self.cpp_standalone = self.is_cpp_standalone()

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
        GSL_support_code += self.make_function_code(self.translate_vector_code(vector_code[None], to_replace))

        # rewrite scalar code, keep variables that are needed in scalar code normal
        # and add variables to dataholder for vector_code
        GSL_main_code += '\n' + self.translate_scalar_code(scalar_code[None],
                                                    variables_in_scalar,
                                                    variables_in_vector)

        scalar_code['GSL'] = GSL_main_code
        kwds['GSL_settings'] = self.method_options
        kwds['support_code_lines'] += GSL_support_code.split('\n')
        kwds['t_array'] = self.get_array_name(self.variables['t']) + '[0]'
        kwds['dt_array'] = self.get_array_name(self.variables['dt']) + '[0]'
        return scalar_code, vector_code, kwds

class GSLCythonCodeGenerator(GSLCodeGenerator):

    syntax = {'end_statement' : '',
              'access_pointer' : '.',
              'start_declare' : 'cdef ',
              'open_function' : ':',
              'open_struct' : ':',
              'end_function' : '',
              'end_struct' : '',
              'open_cast' : '<',
              'close_cast' : '>',
              'diff_var_declaration' : ''}

    def c_data_type(self, dtype):
        return c_data_type(dtype)

    def var_replace_diff_var_lhs(self, var, ind):
        return {'_gsl_{var}_f{ind}'.format(var=var,ind=ind) : 'f[{ind}]'.format(ind=ind)}

    def var_init_lhs(self, var, type):
        return var

    def var_declare(self, type, name, in_struct=False):
        if in_struct:
            return type + ' ' + name
        else:
            return 'cdef ' + type + ' ' + name

    def get_array_name(self, var_obj):
        return self.generator.get_array_name(var_obj)

    def get_pointer_name(self, var_obj):
        return self.get_array_name(var_obj)

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

    syntax = {'end_statement' : ';',
              'access_pointer' : '->',
              'start_declare' : '',
              'open_function' : '\n{',
              'open_struct' : '\n{',
              'end_function' : '\n}',
              'end_struct' : '\n};',
              'open_cast' : '(',
              'close_cast' : ')',
              'diff_var_declaration' : 'const double '}

    def c_data_type(self, dtype):
        return self.generator.c_data_type(dtype)

    def var_replace_diff_var_lhs(self, var, ind):
        f = 'f[{ind}]'.format(ind=ind)
        try:
            if 'unless refractory' in self.variable_flags[var]:
                return {'_gsl_{var}_f{ind}'.format(var=var,ind=ind) : f,
                        'double _gsl_{var}_f{ind};'.format(var=var,ind=ind) : '',
                        'double {f};'.format(f=f) : ''} # in case the replacement of _gsl_var_find to f[ind] happens first
        except KeyError:
            pass
        return {'const double _gsl_{var}_f{ind}'.format(var=var,ind=ind) : f}

    def var_init_lhs(self, var, type):
        return type + var

    def get_array_name(self, var_obj):
        return self.generator.get_array_name(var_obj, access_data=False)

    def get_pointer_name(self, var_obj):
        return self.generator.get_array_name(var_obj, access_data=True)

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
