'''
GSLCodeGenerators for code that uses the ODE solver provided by the GNU Scientific Library (GSL)
'''
import os
import re
import sys

import numpy as np

from brian2.units.fundamentalunits import fail_for_dimension_mismatch
from brian2.core.variables import AuxiliaryVariable, ArrayVariable, Constant
from brian2.core.functions import Function
from brian2.codegen.translation import make_statements
from brian2.codegen.permutation_analysis import (check_for_order_independence,
                                                 OrderDependenceError)
from brian2.core.preferences import prefs, BrianPreference
from brian2.utils.stringtools import get_identifiers, word_substitute
from brian2.parsing.statements import parse_statement
from brian2.codegen.generators import c_data_type


from brian2.core.preferences import PreferenceError

__all__ = ['GSLCodeGenerator', 'GSLWeaveCodeGenerator', 'GSLCythonCodeGenerator']


def valid_gsl_dir(val):
    '''
    Validate given string to be path containing required GSL files.
    '''
    if val is None:
        return True
    if not isinstance(val, basestring):
        raise PreferenceError(('Illegal value for GSL directory: %s, '
                               'has to be str' % (str(val))))
    if not os.path.isdir(val):
        raise PreferenceError(('Illegal value for GSL directory: %s, '
                                'has to be existing directory' % (val)))
    if any(not os.path.isfile(os.path.join(val, 'gsl', filename))
           for filename in ['gsl_odeiv2.h', 'gsl_errno.h', 'gsl_matrix.h']):
        raise PreferenceError(('Illegal value for GSL directory: %s, '
                               'has to contain gsl_odeiv2.h, gsl_errno.h '
                               'and gsl_matrix.h' % (val)))
    return True


prefs.register_preferences(
    'GSL',
    'Directory containing GSL code',
    directory=BrianPreference(
        validator=valid_gsl_dir,
        docs=("Set path to directory containing GSL header files (gsl_odeiv2.h etc.)"
              "\nIf this directory is already in Python's include (e.g. because of "
              "conda installation), this path can be set to None."),
        default=None
    )
)


class GSLCodeGenerator(object):
    '''
    GSL code generator.

    Notes
    -----
    Approach is to first let the already existing code generator for a target
    language do the bulk of the translating from abstract_code to actual code.
    This generated code is slightly adapted to render it GSL compatible.
    The most critical part here is that the vector_code that is normally
    contained in a loop in the ```main()``` is moved to the function ```_GSL_func```
    that is sent to the GSL integrator. The variables used in the vector_code are
    added to a struct named ```dataholder``` and their values are set from the
    Brian namespace just before the scalar code block.
    '''

    def __init__(self, variables, variable_indices, owner, iterate_all,
                 codeobj_class, name, template_name,
                 override_conditional_write=None,
                 allows_scalar_write=False):

        self.generator = codeobj_class.original_generator_class(variables,
                                                                variable_indices,
                                                                owner, iterate_all,
                                                                codeobj_class, name,
                                                                template_name,
                                                                override_conditional_write,
                                                                allows_scalar_write)
        self.method_options = dict(owner.state_updater.method_options)
        self.integrator = owner.state_updater.integrator
        # default timestep to start with is the timestep of the NeuronGroup itself
        self.method_options['dt_start'] = owner.dt.variable.get_value()[0]
        self.variable_flags = owner.state_updater._gsl_variable_flags

    def __getattr__(self, item):
        return getattr(self.generator, item)

    # A series of functions that should be overridden by child class:
    def c_data_type(self, dtype):
        '''
        Get string version of object dtype that is attached to Brian variables. c
        pp_generator already has this function, but the Cython generator does not,
        but we need it for GSL code generation.
        '''
        return NotImplementedError

    def initialize_array(self, varname, values):
        '''
        Initialize a static array with given floating point values. E.g. in C++,
        when called with arguments ``array`` and ``[1.0, 3.0, 2.0]``, this
        method should return ``double array[] = {1.0, 3.0, 2.0}``.

        Parameters
        ----------
        varname : str
            The name of the array variable that should be initialized
        values : list of float
            The values that should be assigned to the array

        Returns
        -------
        code : str
            One or more lines of array initialization code.
        '''
        raise NotImplementedError

    def var_init_lhs(self, var, type):
        '''
        Get string version of the left hand side of an initializing expression

        Parameters
        ----------
        var : str
        type : str

        Returns
        -------
        code : str
            For cpp returns type + var, while for cython just var
        '''
        raise NotImplementedError

    def unpack_namespace_single(self, var_obj, in_vector, in_scalar):
        '''
        Writes the code necessary to pull single variable out of the Brian
        namespace into the generated code.

        The code created is significantly different between cpp and cython,
        so I decided to not make this function general
        over all target languages (i.e. in contrast to most other functions
        that only have syntactical differences)
        '''
        raise NotImplementedError

    # GSL functions that are the same for all target languages:
    def find_function_names(self):
        '''
        Return a list of used function names in the self.variables dictionary

        Functions need to be ignored in the GSL translation process, because the
        brian generator already sufficiently
        dealt with them. However, the brian generator also removes them from the
        variables dict, so there is no
        way to check whether an identifier is a function after the brian
        translation process. This function is called
        before this translation process and the list of function names is stored
        to be used in the GSL translation.

        Returns
        -------
        function_names : list
            list of strings that are function names used in the code
        '''
        variables = self.variables
        return [var for var, var_obj in variables.iteritems()
                if isinstance(var_obj, Function)]

    def is_cpp_standalone(self):
        '''
        Check whether we're running with cpp_standalone.

        Test if `get_device()` is instance `CPPStandaloneDevice`.

        Returns
        -------
        is_cpp_standalone : bool
            whether currently using cpp_standalone device

        See Also
        --------
        is_constant_and_cpp_standalone : uses the returned value
        '''
        # imports here to avoid circular imports
        from brian2.devices.device import get_device
        from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
        device = get_device()
        return isinstance(device, CPPStandaloneDevice)

    def is_constant_and_cpp_standalone(self, var_obj):
        """Check whether self.cpp_standalone and variable is Constant.

        This check is needed because in the case of using the cpp_standalone device we do not
        want to apply our GSL variable conversion (var --> _GSL_dataholder.var), because the cpp_standalone
        code generation process involves replacing constants with their actual value ('freezing').
        This results in code that looks like (if for example var = 1.2): _GSL_dataholder.1.2 = 1.2 and _GSL_dataholder->1.2.
        To prevent repetitive calls to get_device() etc. the outcome of is_cpp_standalone is saved.

        Parameters
        ----------
        var_obj : `Variable`
            instance of brian Variable class describing the variable

        Returns
        -------
        is_cpp_standalone : bool
            whether the used device is cpp_standalone and the given variable is an instance of Constant
        """
        if not hasattr(self, 'cpp_standalone'):
            self.cpp_standalone = self.is_cpp_standalone()
        return isinstance(var_obj, Constant) and self.cpp_standalone

    def find_differential_variables(self, code):
        '''
        Find the variables that were tagged _gsl_{var}_f{ind} and return var, ind pairs.

        `GSLStateUpdater` tagged differential variables and here we extract the information given in these tags.

        Parameters
        ----------
        code : list of strings
            A list of strings containing gsl tagged variables

        Returns
        -------
        diff_vars : dict
            A dictionary with variable names as keys and differential equation index as value
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
        Add differential variable-related strings that need to be replaced to go
        from normal brian to GSL code

        From the code generated by Brian's 'normal' generators (cpp_generator or
        cython_generator a few bits of text need to be replaced to get GSL
        compatible code. The bits of text related to differential equation
        variables are put in the replacer dictionary in this function.

        Parameters
        ----------
        diff_vars : dict
            dictionary with variables as keys and differential equation index as value

        Returns
        -------
        to_replace : dict
            dictionary with strings that need to be replaced as keys and the
            strings that will replace them as values
        '''
        variables = self.variables
        to_replace = {}
        for var, diff_num in diff_vars.items():
            to_replace.update(self.var_replace_diff_var_lhs(var, diff_num))
            var_obj = variables[var]
            array_name = self.generator.get_array_name(var_obj, access_data=True)
            idx_name = '_idx' #TODO: could be dynamic?
            replace_what = '{var} = {array_name}[{idx_name}]'.format(array_name=array_name, idx_name=idx_name, var=var)
            replace_with = '{var} = _GSL_y[{ind}]'.format(ind=diff_num, var=var)
            to_replace[replace_what] = replace_with
        return to_replace

    def get_dimension_code(self, diff_num):
        '''
        Generate code for function that sets the dimension of the ODE system.

        GSL needs to know how many differential variables there are in the
        ODE system. Since the current approach is to have the code in the vector
        loop the same for all simulations, this dimension is set by an external
        function. The code for this set_dimension functon is written here.
        It is assumed the code will be the same for each target language with the
        exception of some syntactical differences

        Parameters
        ----------
        diff_num : int
            Number of differential variables that describe the ODE system

        Returns
        -------
        set_dimension_code : str
            The code describing the target language function in a single string
        '''
        code = ['\n{start_declare}int set_dimension(size_t * dimension){open_function}']
        code += ['\tdimension[0] = %d{end_statement}'%diff_num]
        code += ['\treturn GSL_SUCCESS{end_statement}{end_function}']
        return ('\n').join(code).format(**self.syntax)

    def yvector_code(self, diff_vars):
        '''
        Generate code for function dealing with GSLs y vector.

        The values of differential variables have to be transferred from
        Brian's namespace to a vector that is given to GSL. The transferring
        from Brian --> y and back from y --> Brian after integration happens in
        separate functions. The code for these is written here.

        Parameters
        ----------
        diff_vars : dictionary
            Dictionary containing variable names as keys (str) and differential
            variable index as value

        Returns
        -------
        yvector_code : str
            The code for the two functions (``_fill_y_vector`` and
            ``_empty_y_vector``) as single string.
        '''
        fill_y = [("\n{start_declare}int _fill_y_vector(_dataholder *"
                  "_GSL_dataholder, double * _GSL_y, int _idx){open_function}")]
        empty_y = [("\n{start_declare}int _empty_y_vector(_dataholder * "
                    "_GSL_dataholder, double * _GSL_y, int _idx){"
                    "open_function}")]
        for var, diff_num in diff_vars.items():
            diff_num = int(diff_num)
            array_name = self.generator.get_array_name(self.variables[var],
                                                       access_data=True)
            fill_y += [("\t_GSL_y[%d] = _GSL_dataholder{access_pointer}%s["
                        "_idx]{end_statement}"%(diff_num, array_name))]
            empty_y += [("\t_GSL_dataholder{access_pointer}%s[_idx] = _GSL_y["
                        "%d]{end_statement}"%(array_name, diff_num))]
        fill_y += ['\treturn GSL_SUCCESS{end_statement}{end_function}']
        empty_y += ['\treturn GSL_SUCCESS{end_statement}{end_function}']
        return ('\n').join(fill_y + empty_y).format(**self.syntax)

    def make_function_code(self, lines):
        '''
        Add lines of GSL translated vector code to 'non-changing' _GSL_func code.

        Adds nonchanging aspects of GSL _GSL_func code to lines of code
        written somewhere else (`translate_vector_code`). Here these lines
        are put between the non-changing parts of the code and the
        target-language specific syntax is added.

        Parameters
        ----------
        lines : str
            Code containing GSL version of equations

        Returns
        -------
        function_code : str
            code describing ``_GSL_func`` that is sent to GSL integrator.
        '''
        code = [("\n{start_declare}int _GSL_func(double t, const double "
                "_GSL_y[], double f[], void * params){open_function}"
                "\n\t{start_declare}_dataholder * _GSL_dataholder = {open_cast}"
                "_dataholder *{close_cast} params{end_statement}"
                "\n\t{start_declare}int _idx = _GSL_dataholder{access_pointer}_idx"
                "{end_statement}")]
        code += [lines]
        code += ["\treturn GSL_SUCCESS{end_statement}{end_function}"]
        return ('\n').join(code).format(**self.syntax)

    def write_dataholder_single(self, var_obj):
        '''
        Return string declaring a single variable in the ``_dataholder`` struct.

        Parameters
        ----------
        var_obj : `Variable`

        Returns
        -------
        code : str
            string describing this variable object as required for the ``_dataholder`` struct
            (e.g. ``double* _array_neurongroup_v``)
        '''
        dtype = self.c_data_type(var_obj.dtype)
        if isinstance(var_obj, ArrayVariable):
            pointer_name = self.get_array_name(var_obj, access_data=True)
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
        Return string with full code for _dataholder struct.

        Parameters
        ----------
        variables_in_vector : dict
            dictionary containing variable name as key and `Variable` as value

        Returns
        -------
        code : str
            code for _dataholder struct
        '''
        code = ['\n{start_declare}struct _dataholder{open_struct}']
        code += ['\tint _idx{end_statement}']
        for var, var_obj in variables_in_vector.items():
            if var == 't' or '_gsl' in var or self.is_constant_and_cpp_standalone(var_obj):
                continue
            code += ['\t'+self.write_dataholder_single(var_obj)]
        code += ['{end_struct}']
        return ('\n').join(code).format(**self.syntax)

    def scale_array_code(self, diff_vars, method_options):
        '''
        Return code for definition of ``_GSL_scale_array`` in generated code.

        Parameters
        ----------
        diff_vars : dict
            dictionary with variable name (str) as key and differential variable
            index (int) as value
        method_options : dict
            dictionary containing integrator settings

        Returns
        -------
        code : str
            full code describing a function returning a array containing doubles
            with the absolute errors for each differential variable (according
            to their assigned index in the GSL StateUpdater)
        '''
        # get scale values per variable from method_options
        abs_per_var = method_options['absolute_error_per_variable']
        abs_default = method_options['absolute_error']

        if not isinstance(abs_default, float):
            raise TypeError(("The absolute_error key in method_options should be "
                             "a float. Was type %s" % (str(type(abs_default)))))

        if abs_per_var is None:
            diff_scale = {var: float(abs_default) for var in diff_vars.keys()}
        elif isinstance(abs_per_var, dict):
            diff_scale = {}
            for var, error in abs_per_var.items():
                # first do some checks on input
                if not var in diff_vars:
                    if not var in self.variables:
                        raise KeyError("absolute_error specified for variable "
                                       "that does not exist: %s"%var)
                    else:
                        raise KeyError("absolute_error specified for variable "
                                       "that is not being integrated: %s"%var)
                fail_for_dimension_mismatch(error, self.variables[var],
                                            ("Unit of absolute_error_per_variable "
                                             "for variable %s does not match "
                                             "unit of varialbe itself"%var))
                # if all these are passed we can add the value for error in base units
                diff_scale[var] = float(error)
            # set the variables that are not mentioned to default value
            for var in diff_vars.keys():
                if var not in abs_per_var:
                    diff_scale[var] = float(abs_default)
        else:
            raise TypeError(("The absolute_error_per_variable key in method_options "
                             "should either be None or a dictionary "
                             "containing the error for each individual state variable. "
                             "Was type %s"%(str(type(abs_per_var)))))
        # write code
        return self.initialize_array('_GSL_scale_array', [diff_scale[var] for var in sorted(diff_vars)])

    def find_undefined_variables(self, statements):
        '''
        Find identifiers that are not in ``self.variables`` dictionary.

        Brian does not save the ``_lio_`` variables it uses anywhere. This is
        problematic for our GSL implementation because we save the lio variables
        in the ``_dataholder`` struct (for which we need the datatype of the
        variables). This function adds the left hand side variables that are
        used in the vector code to the variable dictionary as
        `AuxiliaryVariable`\ s (all we need later is the datatype).

        Parameters
        ----------
        statements : list
            list of statement objects (need to have the dtype attribute)

        Notes
        -----
        I keep ``self.variables`` and ``other_variables`` separate so I can
        distinguish what variables are in the Brian namespace and which ones are
        defined in the code itself.
        '''
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
        Find all the variables used in the right hand side of the given
        expressions.

        Parameters
        ----------
        statements : list
            list of statement objects

        Returns
        -------
        used_variables : dict
            dictionary of variables that are used as variable name (str),
            `Variable` pairs.
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
                used_variables[var] = var_obj # save as object because this has
                                              # all needed info (dtype, name, isarray)

        # I don't know a nicer way to do this, the above way misses write
        # variables (e.g. not_refractory)..
        read, write, _ = self.array_read_write(statements)
        for var in (read|write):
            if var not in used_variables:
                used_variables[var] = variables[var] # will always be array and
                                                     # thus exist in variables

        return used_variables

    def to_replace_vector_vars(self, variables_in_vector, ignore=frozenset()):
        '''
        Create dictionary containing key, value pairs with to be replaced text
        to translate from conventional Brian to GSL.

        Parameters
        ----------
        variables_in_vector : dict
            dictionary with variable name (str), `Variable` pairs of variables
            occurring in vector code
        ignore : set, optional
            set of strings with variable names that should be ignored

        Returns
        -------
        to_replace : dict
            dictionary with strings that need to be replaced i.e. _lio_1 will be
            _GSL_dataholder._lio_1 (in cython) or _GSL_dataholder->_lio_1 (cpp)

        Notes
        -----
        t will always be added because GSL defines its own t.
        i.e. for cpp: {'const t = _ptr_array_defaultclock_t[0];' : ''}
        '''
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
                # does not have to be processed by GSL generator
                self.variables_to_be_processed.remove(var_obj.name)
                continue
            if isinstance(var_obj, ArrayVariable):
                pointer_name = self.get_array_name(var_obj, access_data=True)
                to_replace[pointer_name] = '_GSL_dataholder' + access_pointer + pointer_name
            else:
                to_replace[var] = '_GSL_dataholder' + access_pointer + var

        # also make sure t declaration is replaced if in code
        if t_in_code is not None:
            t_declare = self.var_init_lhs('t', 'const double ')
            array_name = self.get_array_name(t_in_code, access_data=True)
            end_statement = self.syntax['end_statement']
            replace_what = ("{t_declare} = {array_name}[0]"
                            "{end_statement}".format(t_declare=t_declare,
                                                     array_name=array_name,
                                                     end_statement=end_statement))
            to_replace[replace_what] = ''
            self.variables_to_be_processed.remove('t')

        return to_replace

    def unpack_namespace(self, variables_in_vector, variables_in_scalar,
                         ignore=frozenset()):
        '''
        Write code that unpacks Brian namespace to cython/cpp namespace.

        For vector code this means putting variables in _dataholder (i.e.
        _GSL_dataholder->var or _GSL_dataholder.var = ...)
        Note that code is written so a variable could occur both in scalar and
        vector code

        Parameters
        ----------
        variables_in_vector : dict
            dictionary with variable name (str), `Variable` pairs of variables
            occurring in vector code
        variables_in_scalar : dict
            dictionary with variable name (str), `Variable` pairs of variables
             occurring in scalar code
        ignore : set, optional
            set of string names of variables that should be ignored

        Returns
        -------
        unpack_namespace_code : str
            code fragment unpacking the Brian namespace (setting variables in
            the _dataholder struct in case of vector)
        '''
        code = []
        for var, var_obj in self.variables.items():
            if var in ignore:
                continue
            if self.is_constant_and_cpp_standalone(var_obj):
                continue
            in_vector = var in variables_in_vector
            in_scalar = var in variables_in_scalar
            if in_vector:
                self.variables_to_be_processed.remove(var)
            code += [self.unpack_namespace_single(var_obj, in_vector, in_scalar)]
        return ('\n').join(code)

    def translate_vector_code(self, code_lines, to_replace):
        '''
        Translate vector code to GSL compatible code by substituting fragments
        of code.

        Parameters
        ----------
        code_lines : list
            list of strings describing the vector_code
        to_replace: dict
            dictionary with to be replaced strings (see to_replace_vector_vars
            and to_replace_diff_vars)

        Returns
        -------
        vector_code : str
            New code that is now to be added to the function that is sent to the
            GSL integrator
        '''
        code = []
        for expr_set in code_lines:
            for line in expr_set.split('\n'): # every line seperate to make tabbing correct
                code += ['\t' + line]
        code = ('\n').join(code)
        code = word_substitute(code, to_replace)

        # special substitute because of limitations of regex word boundaries with
        # variable[_idx]
        for from_sub, to_sub in to_replace.items():
            m = re.search('\[(\w+)\];?$', from_sub)
            if m:
                code = re.sub(re.sub('\[','\[', from_sub), to_sub, code)

        if '_gsl' in code:
            raise AssertionError(('Translation failed, _gsl still in code (should only '
                             'be tag, and should be replaced.\n'
                             'Code:\n%s' % code))

        return code

    def translate_scalar_code(self, code_lines, variables_in_scalar,
                              variables_in_vector):
        '''
        Translate scalar code: if calculated variables are used in the vector_code
        their value is added to the variable in the _dataholder.

        Parameters
        ----------
        code_lines : list
            list of strings containing scalar code
        variables_in_vector : dict
            dictionary with variable name (str), `Variable` pairs of variables
            occurring in vector code
        variables_in_scalar : dict
            dictionary with variable name (str), `Variable` pairs of variables
            occurring in scalar code

        Returns
        -------
        scalar_code : str
            code fragment that should be injected in the main before the loop
        '''
        code = []
        for line in code_lines:
            m = re.search('(\w+ = .*)', line)
            try:
                new_line = m.group(1)
                var, op, expr, comment = parse_statement(new_line)
            except (ValueError, AttributeError):
                code += [line]
                continue
            if var in variables_in_scalar.keys():
                code += [line]
            elif var in variables_in_vector.keys():
                if var == 't':
                    continue
                try:
                    self.variables_to_be_processed.remove(var)
                except KeyError:
                    raise AssertionError(("Trying to process variable named %s by "
                                          "putting its value in the _GSL_dataholder "
                                          "based on scalar code, but the variable "
                                          "has been processed already." % var))
                code += ['_GSL_dataholder.{var} {op} {expr} {comment}'.format(
                        var=var, op=op, expr=expr, comment=comment)]
        return '\n'.join(code)

    def add_gsl_variables_as_non_scalar(self, diff_vars):
        '''
        Add _gsl variables as non-scalar.

        In `GSLStateUpdater` the differential equation variables are substituted
        with GSL tags that describe the information needed to translate the
        conventional Brian code to GSL compatible code. This function tells
        Brian that the variables that contain these tags should always be vector
        variables. If we don't do this, Brian renders the tag-variables as
        scalar if no vector variables are used in the right hand side of the
        expression.

        Parameters
        ----------
        diff_vars : dict
            dictionary with variables as keys and differential equation index as
            value
        '''
        for var, ind in diff_vars.items():
            name = '_gsl_{var}_f{ind}'.format(var=var,ind=ind)
            self.variables[name] = AuxiliaryVariable(var, scalar=False)

    def add_meta_variables(self, options):
        if options['use_last_timestep']:
            try:
                N = int(self.variables['N'].get_value())
                self.owner.variables.add_array('_last_timestep', size=N,
                                               values=np.ones(N)*options['dt_start'],
                                               dtype=np.float64)
            except KeyError:
                # has already been run
                pass
            self.variables['_last_timestep'] = self.owner.variables.get('_last_timestep')
            pointer_last_timestep = self.get_array_name(self.variables['_last_timestep'])+ '[_idx]'
        else:
            pointer_last_timestep = None

        if options['save_failed_steps']:
            N = int(self.variables['N'].get_value())
            try:
                self.owner.variables.add_array('_failed_steps', size=N, dtype=np.int32)
            except KeyError:
                # has already been run
                pass
            self.variables['_failed_steps'] = self.owner.variables.get('_failed_steps')
            pointer_failed_steps = self.get_array_name(self.variables['_failed_steps']) + '[_idx]'
        else:
            pointer_failed_steps = None

        if options['save_step_count']:
            N = int(self.variables['N'].get_value())
            try:
                self.owner.variables.add_array('_step_count', size=N, dtype=np.int32)
            except KeyError:
                # has already been run
                pass
            self.variables['_step_count'] = self.owner.variables.get('_step_count')
            pointer_step_count = self.get_array_name(self.variables['_step_count']) + '[_idx]'
        else:
            pointer_step_count = None

        return {'pointer_last_timestep' : pointer_last_timestep,
                'pointer_failed_steps' : pointer_failed_steps,
                'pointer_step_count' : pointer_step_count}

    def translate(self, code, dtype): # TODO: it's not so nice we have to copy the contents of this function..
        '''
        Translates an abstract code block into the target language.
        '''
        # first check if user code is not using variables that are also used by GSL
        reserved_variables = ['_dataholder', '_fill_y_vector', '_empty_y_vector',
                              '_GSL_dataholder', '_GSL_y', '_GSL_func']
        if any([var in self.variables for var in reserved_variables]):
            # import here to avoid circular import
            raise ValueError(("The variables %s are reserved for the GSL "
                              "internal code."%(str(reserved_variables))))

        # if the following statements are not added, Brian translates the
        # differential expressions in the abstract code for GSL to scalar statements
        # in the case no non-scalar variables are used in the expression
        diff_vars = self.find_differential_variables(code.values())
        self.add_gsl_variables_as_non_scalar(diff_vars)

        # add arrays we want to use in generated code before self.generator.translate() so
        # brian does namespace unpacking for us
        pointer_names = self.add_meta_variables(self.method_options)

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
                if self.has_repeated_indices(vs):     # only do order dependence if there are repeated indices
                    check_for_order_independence(vs,
                                                 self.generator.variables,
                                                 self.generator.variable_indices)
            except OrderDependenceError:
                # If the abstract code is only one line, display it in ful   l
                if len(vs) <= 1:
                    error_msg = 'Abstract code: "%s"\n' % vs[0]
                else:
                    error_msg = ('%_GSL_driver lines of abstract code, first line is: '
                                 '"%s"\n') % (len(vs), vs[0])

        # save function names because self.generator.translate_statement_sequence
        # deletes these from self.variables but we need to know which identifiers
        # we can safely ignore (i.e. we can ignore the functions because they are
        # handled by the original generator)
        self.function_names = self.find_function_names()

        scalar_code, vector_code, kwds = self.generator.translate_statement_sequence(scalar_statements,
                                                                                     vector_statements)

        ############ translate code for GSL

        # first check if any indexing other than '_idx' is used (currently not supported)
        for code_list in scalar_code.values()+vector_code.values():
            for code in code_list:
                m = re.search('\[(\w+)\]', code)
                if m is not None:
                    if m.group(1) != '0' and m.group(1) != '_idx':
                        from brian2.stateupdaters.base import UnsupportedEquationsException
                        raise UnsupportedEquationsException(("Equations result in state "
                                                             "updater code with indexing "
                                                             "other than '_idx', which "
                                                             "is currently not supported "
                                                             "in combination with the "
                                                             "GSL stateupdater."))

        # differential variable specific operations
        to_replace = self.diff_var_to_replace(diff_vars)
        GSL_support_code = self.get_dimension_code(len(diff_vars))
        GSL_support_code += self.yvector_code(diff_vars)

        # analyze all needed variables; if not in self.variables: put in separate dic.
        # also keep track of variables needed for scalar statements and vector statements
        other_variables = self.find_undefined_variables(scalar_statements[None] +
                                                        vector_statements[None])
        variables_in_scalar = self.find_used_variables(scalar_statements[None],
                                                       other_variables)
        variables_in_vector = self.find_used_variables(vector_statements[None],
                                                       other_variables)
        # so that _dataholder holds diff_vars as well, even if they don't occur
        # in the actual statements
        for var in diff_vars.keys():
            if not var in variables_in_vector:
                variables_in_vector[var] = self.variables[var]
        # lets keep track of the variables that eventually need to be added to
        # the _GSL_dataholder somehow
        self.variables_to_be_processed = variables_in_vector.keys()

        # add code for _dataholder struct
        GSL_support_code = self.write_dataholder(variables_in_vector) + GSL_support_code
        # add e.g. _lio_1 --> _GSL_dataholder._lio_1 to replacer
        to_replace.update(self.to_replace_vector_vars(variables_in_vector,
                                                      ignore=diff_vars.keys()))
        # write statements that unpack (python) namespace to _dataholder struct
        # or local namespace
        GSL_main_code = self.unpack_namespace(variables_in_vector, variables_in_scalar, ['t'])

        # rewrite actual calculations described by vector_code and put them in _GSL_func
        func_code = self.translate_one_statement_sequence(vector_statements[None],
                                                          scalar=False)
        GSL_support_code += self.make_function_code(self.translate_vector_code(func_code,
                                                                               to_replace))
        scalar_func_code = self.translate_one_statement_sequence(scalar_statements[None],
                                                                 scalar=True)
        # rewrite scalar code, keep variables that are needed in scalar code normal
        # and add variables to _dataholder for vector_code
        GSL_main_code += '\n' + self.translate_scalar_code(scalar_func_code,
                                                    variables_in_scalar,
                                                    variables_in_vector)
        if len(self.variables_to_be_processed) > 0:
            raise AssertionError(("Not all variables that will be used in the vector "
                                  "code have been added to the _GSL_dataholder. This "
                                  "might mean that the _GSL_func is using unitialized "
                                  "variables."
                                  "\nThe unprocessed variables "
                                  "are: %s" % (str(self.variables_to_be_processed))))

        scalar_code['GSL'] = GSL_main_code
        kwds['define_GSL_scale_array'] = self.scale_array_code(diff_vars,
                                                               self.method_options)
        kwds['n_diff_vars'] = len(diff_vars)
        kwds['GSL_settings'] = dict(self.method_options)
        kwds['GSL_settings']['integrator'] = self.integrator
        kwds['support_code_lines'] += GSL_support_code.split('\n')
        kwds['t_array'] = self.get_array_name(self.variables['t']) + '[0]'
        kwds['dt_array'] = self.get_array_name(self.variables['dt']) + '[0]'
        kwds['define_dt'] = 'dt' not in variables_in_scalar
        kwds['cpp_standalone'] = self.is_cpp_standalone()
        for key, value in pointer_names.items():
            kwds[key] = value
        return scalar_code, vector_code, kwds


class GSLCythonCodeGenerator(GSLCodeGenerator):

    syntax = {'end_statement': '',
              'access_pointer': '.',
              'start_declare': 'cdef ',
              'open_function': ':',
              'open_struct': ':',
              'end_function': '',
              'end_struct': '',
              'open_cast': '<',
              'close_cast': '>',
              'diff_var_declaration': ''}

    def c_data_type(self, dtype):
        return c_data_type(dtype)

    def initialize_array(self, varname, values):
        value_list = ', '.join(repr(v) for v in values)
        code = 'cdef double {varname}[{n_values}]\n'
        code += '{varname}[:] = [{value_list}]'
        return code.format(varname=varname, value_list=value_list,
                           n_values=len(values))

    def var_replace_diff_var_lhs(self, var, ind):
        return {'_gsl_{var}_f{ind}'.format(var=var, ind=ind):
                    'f[{ind}]'.format(ind=ind)}

    def var_init_lhs(self, var, type):
        return var

    def unpack_namespace_single(self, var_obj, in_vector, in_scalar):
        code = []
        if isinstance(var_obj, ArrayVariable):
            array_name = self.generator.get_array_name(var_obj)
            dtype = self.c_data_type(var_obj.dtype)
            if in_vector:
                code += [('_GSL_dataholder.{array} = <{dtype} *> '
                          '_buf_{array}.data'.format(array=array_name, dtype=dtype))]
            if in_scalar:
                code += [('{array} = <{dtype} *> '
                          '_buf_{array}.data'.format(array=array_name, dtype=dtype))]
        else:
            if in_vector:
                code += ['_GSL_dataholder.{var} = _namespace["{var}"]'.format(var=var_obj.name)]
            if in_scalar:
                code += ['{var} = _namespace["{var}"]'.format(var=var_obj.name)]
        return '\n'.join(code)

    @staticmethod
    def get_array_name( var, access_data=True):
        # We have to do the import here to avoid circular import dependencies.
        from brian2.codegen.generators.cython_generator import CythonCodeGenerator
        return CythonCodeGenerator.get_array_name(var, access_data)


class GSLWeaveCodeGenerator(GSLCodeGenerator):

    def __getattr__(self, item):
        return getattr(self.generator, item)

    syntax = {'end_statement': ';',
              'access_pointer': '->',
              'start_declare': '',
              'open_function': '\n{',
              'open_struct' :'\n{',
              'end_function': '\n}',
              'end_struct': '\n};',
              'open_cast': '(',
              'close_cast': ')',
              'diff_var_declaration': 'const scalar '}

    def c_data_type(self, dtype):
        return self.generator.c_data_type(dtype)

    def initialize_array(self, varname, values):
        value_list = ', '.join(repr(v) for v in values)
        return 'double const %s[] = {%s};' % (varname, value_list)

    def var_replace_diff_var_lhs(self, var, ind):
        scalar_dtype = self.c_data_type(prefs.core.default_float_dtype)
        f = 'f[{ind}]'.format(ind=ind)
        try:
            if 'unless refractory' in self.variable_flags[var]:
                return {'_gsl_{var}_f{ind}'.format(var=var,ind=ind) : f,
                        '{scalar_dtype} _gsl_{var}_f{ind};'.format(var=var, ind=ind,
                                                                   scalar_dtype=scalar_dtype): '',
                        '{scalar_dtype} {f};'.format(f=f, scalar_dtype=scalar_dtype): ''} # in case the replacement
                                                                                          # of _gsl_var_find to f[ind] happens first
        except KeyError:
            pass
        return {'const {scalar_dtype} _gsl_{var}_f{ind}'.format(scalar_dtype=scalar_dtype,
                                                                var=var, ind=ind) : f}

    def var_init_lhs(self, var, type):
        return type + var

    def unpack_namespace_single(self, var_obj, in_vector, in_scalar):
        if isinstance(var_obj, ArrayVariable):
            pointer_name = self.get_array_name(var_obj, access_data=True)
            array_name = self.get_array_name(var_obj)
            if in_vector:
                return '_GSL_dataholder.{ptr} = {array};'.format(ptr=pointer_name,
                                                                 array=array_name)
            else:
                return ''
        else:
            if in_vector:
                return '_GSL_dataholder.{var} = {var};'.format(var=var_obj.name)
            else:
                return ''

    @staticmethod
    def get_array_name( var, access_data=True):
        # We have to do the import here to avoid circular import dependencies.
        from brian2.codegen.runtime.weave_rt import WeaveCodeGenerator
        return WeaveCodeGenerator.get_array_name(var, access_data)
