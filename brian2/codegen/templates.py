'''
Handles loading templates from a directory.
'''
import re
from collections import Mapping

from jinja2 import (Environment, PackageLoader, ChoiceLoader, StrictUndefined,
                    TemplateNotFound)

from brian2.utils.stringtools import (indent, strip_empty_lines,
                                      get_identifiers, word_substitute)

from brian2.core.variables import ArrayVariable, Constant, AuxiliaryVariable
from brian2.core.functions import Function
from brian2.parsing.statements import parse_statement

from brian2.core.preferences import prefs

# for some reason I get an error with importing get_cpp_dtype (ImportError: cannot import name NumpyCodeObject)
from brian2.core.variables import get_dtype_str
# from brian2.codegen.generators.cython_generator import data_type_conversion_table, cpp_dtype, get_cpp_dtype
# so I just redefined it TODO: solve this
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
def get_cpp_dtype(obj):
    return cpp_dtype[get_dtype_str(obj)]

__all__ = ['Templater']

AUTOINDENT_START = '%%START_AUTOINDENT%%'
AUTOINDENT_END = '%%END_AUTOINDENT%%'


def autoindent(code):
    if isinstance(code, list):
        code = '\n'.join(code)
    if not code.startswith('\n'):
        code = '\n'+code
    if not code.endswith('\n'):
        code = code + '\n'
    return AUTOINDENT_START+code+AUTOINDENT_END


def autoindent_postfilter(code):
    lines = code.split('\n')
    outlines = []
    addspaces = 0
    for line in lines:
        if AUTOINDENT_START in line:
            if addspaces>0:
                raise SyntaxError("Cannot nest autoindents")
            addspaces = line.find(AUTOINDENT_START)
            line = line.replace(AUTOINDENT_START, '')
        if AUTOINDENT_END in line:
            line = line.replace(AUTOINDENT_END, '')
            addspaces = 0
        outlines.append(' '*addspaces+line)
    return '\n'.join(outlines)


def variables_to_array_names(variables, access_data=True):
    from brian2.devices.device import get_device
    device = get_device()
    names = [device.get_array_name(var, access_data=access_data)
             for var in variables]
    return names

def find_variable_mapping(generator=None, *variabledicts):
    variable_mapping = {}
    to_array = {}
    for variables in variabledicts:
        for key, value in variables.iteritems():
            if isinstance(value, ArrayVariable):
                to_array[key] = value
        for var, array_var in zip(to_array.keys(), variables_to_array_names(to_array.values())):
            variable_mapping[var] = array_var
    return variable_mapping

def find_datatypes(*variabledicts):
    datatypes = {}
    for variables in variabledicts:
        for key, value in variables.iteritems():
            try:
                datatypes[key] = get_cpp_dtype(value)
            except KeyError:
                pass
    return datatypes

def find_differential_variables(code):
    diff_vars = {}
    for expr_set in code:
        for expr in expr_set.split('\n'):
            expr = expr.strip(' ')
        try:
            lhs, op, rhs, comment = parse_statement(expr)
        except ValueError:
            pass
        m = re.match('[const double ]*_gsl_(.+?)_f([0-9]*)', lhs)
        if m:
            diff_vars[m.group(1)] = m.group(2)
    return diff_vars

def write_GSL_support_code(vector_code, variables, extra_information):

    other_variables = extra_information['other_variables']
    datatypes = find_datatypes(variables, other_variables)
    variable_mapping = extra_information['variable_mapping']
    diff_vars = find_differential_variables(vector_code)

    struct_parameters = ['\nstruct parameters\n{',
                         '\tint _idx;']
    set_dimension = ['\nint set_dimension(size_t * dimension)\n{']
    allocate_y_vector = ['\ndouble * assign_memory_y()\n{']
    func_fill_yvector = ['\nint fill_y_vector(parameters * p, double * y, int _idx)\n{']
    func_empty_yvector = ['\nint empty_y_vector(parameters * p, double * y, int _idx)\n{']
    func_begin = ['\nint func(double t, const double y[], double f[], void * params)\n{',
            '\tstruct parameters * p = (struct parameters *)params;',
            '\tint _idx = p->_idx;']
    func_end = []

    defined = ['t', 'dt']
    to_replace = {}
    parameters = {} # variables we want in parameters statevars
    func_declarations = {} # declarations that go in beginning of function (cdef double v, cdef double spike etc.)

    temp_func_a = ['''
    int temp_func_a(parameters * p, int _idx)
    {
        p->a = p->_ptr_array_neurongroup_a[_idx];
        return GSL_SUCCESS;
    }
    ''']
    temp_defined_a = ['a']
    struct_parameters += ['\tdouble a;']
    struct_parameters += ['\tdouble* __restrict _ptr_array_neurongroup_a;']
    to_replace['a'] = 'p->a'

    for var in diff_vars:
        diff_num = diff_vars[var]
        array_name = variable_mapping[var]['pointer']
        to_replace['const double _gsl_{var}_f{ind}'.format(var=var, ind=diff_num)] = 'f[{ind}]'.format(ind=diff_num)
        to_replace['const double _gsl_{var}_y{ind}'.format(var=var, ind=diff_num)] = 'y[{ind}]'.format(ind=diff_num)
        func_fill_yvector += ['\ty[{ind}] = p->{var}[_idx];'.format(ind=diff_num, var=array_name)]
        func_empty_yvector += ['\tp->{var}[_idx] = y[{ind}];'.format(ind=diff_num, var=array_name)]

    for var in extra_information['vector_variables']:
        if var in defined or var in temp_defined_a:
            continue
        if var in variable_mapping:
            array_name = variable_mapping[var]['pointer']
            parameters[var] = '\t{dt}* {res} {name};'.format(dt=datatypes[var],
                                                             res=variable_mapping[var]['restrict'],
                                                             name=array_name)
            to_replace[array_name] = 'p->' + array_name
            defined += [var]
        else:
            parameters[var] = '\t{datatype} {var};'.format(datatype=datatypes[var], var=var)
            to_replace[var] = 'p->' + var
            defined += [var]

    for expr_set in vector_code:
        for expr in expr_set.split('\n'):
            try:
                lhs, op, rhs, comment = parse_statement(expr)
                lhs = lhs.replace('const double ', '')
            except ValueError: # if statements?
                func_end += ['\t'+expr]
                continue
            if lhs in temp_defined_a:
                continue
            if (lhs in diff_vars and variable_mapping[lhs]['pointer'] in rhs) or (rhs in diff_vars and variable_mapping[rhs]['pointer'] in lhs):
                func_end += ['\tconst double {var} = y[{ind}];'.format(var=lhs, ind=diff_vars[lhs])]
                continue
            if lhs in ['t', 'dt', 't1']: # ignore t = _array_defaultclock_t[0]
                continue

            func_end += ['\t'+word_substitute(expr, to_replace)]

    for name, expr in parameters.iteritems():
        struct_parameters += [expr]
    struct_parameters += ['};']

    for name, expr in func_declarations.iteritems():
        func_begin += [expr]

    func_end += ['\treturn GSL_SUCCESS;\n}']

    func_fill_yvector += ['\treturn GSL_SUCCESS;\n}']
    func_empty_yvector += ['\treturn GSL_SUCCESS;\n}']

    allocate_y_vector += ['\treturn (double *)malloc({diff_num}*sizeof(double));\n'.format(diff_num=len(diff_vars.keys()))+'}']

    set_dimension += ['\tdimension[0] = {diff_num};\n'.format(diff_num=len(diff_vars.keys()))+'\treturn GSL_SUCCESS;\n}']

    everything = struct_parameters + set_dimension + allocate_y_vector + temp_func_a +\
                 func_fill_yvector + func_empty_yvector + func_begin + func_end
    print ('\n').join(everything)
    return everything

def temp_write_GSL_support_code(vector_code, variables, other_variables, variables_needed):
    #TODO: another way would be to just rewrite an only vector_code dictionary, if it's not a problem to have double entries to variables?
    #TODO: Handle refractoriness differently
    '''
    This function translates the vector_code to GSL code including the definition of the parameter struct and fill_y_vector etc.
    It does so based on the statements sent to the Templater, and infers what is needed.
    '''
    datatypes = find_datatypes(variables, other_variables)
    variable_mapping = find_variable_mapping(variables)
    diff_vars = find_differential_variables(vector_code)

    struct_parameters = ['\ncdef struct parameters:',
                         '\tint _idx']
    set_dimension = ['\ncdef int set_dimension(size_t * dimension):']
    allocate_y_vector = ['\ncdef double* assign_memory_y():']
    func_fill_yvector = ['\ncdef int fill_y_vector(parameters * p, double * y, int _idx):']
    func_empty_yvector = ['\ncdef int empty_y_vector(parameters * p, double * y, int _idx):']
    func_begin = ['\ncdef int func(double t, const double y[], double f[], void * params):',
            '\tcdef parameters * p = <parameters *>params',
            '\tcdef int _idx = p._idx']
    func_end = []

    defined = ['t']
    to_replace = {}
    parameters = {} # variables we want in parameters statevars
    func_declarations = {} # declarations that go in beginning of function (cdef double v, cdef double spike etc.)

    for var in variables_needed:
        if var in defined:
            continue
        if var in diff_vars:
            array_name = variable_mapping[var]
            to_replace['_gsl_{var}_f{ind}'.format(var=var, ind=diff_vars[var])] = 'f[{ind}]'.format(ind=diff_vars[var])
            to_replace['_gsl_{var}_y{ind}'.format(var=var, ind=diff_vars[var])] = 'y[{ind}]'.format(ind=diff_vars[var])
            func_declarations[var] = '\tcdef {datatype} {var}'.format(datatype=datatypes[var], var=var)
            parameters[var] = '\t{datatype} * {var}'.format(datatype=datatypes[var], var=array_name)
            func_fill_yvector += ['\ty[{ind}] = p.{var}[_idx]'.format(ind=diff_vars[var], var=array_name)]
            func_empty_yvector += ['\tp.{var}[_idx] = y[{ind}]'.format(ind=diff_vars[var], var=array_name)]
        elif var in variable_mapping:
            array_name = variable_mapping[var]
            func_declarations[var] = '\t{datatype} {var}'.format(datatype=datatypes[var], var=var)
            parameters[var] = '\t{datatype} * {var}'.format(datatype=datatypes[var], var=array_name)
            to_replace[array_name] = 'p.'+array_name
        else:
            to_replace[var] = 'p.' + var
            parameters[var] = '\t{datatype} {var}'.format(datatype=datatypes[var], var=var)

    for expr_set in vector_code:
        for expr in expr_set.split('\n'):
            try:
                lhs, op, rhs, comment = parse_statement(expr)
            except ValueError: # if statements?
                func_end += ['\t'+expr]
                continue
            if (lhs in diff_vars and variable_mapping[lhs] in rhs) or (rhs in diff_vars and variable_mapping[rhs] in lhs):
                func_end += ['\t{var} = y[{ind}]'.format(var=lhs, ind=diff_vars[lhs])]
                continue
            if lhs in defined: # ignore t = _array_defaultclock_t[0]
                continue

            func_end += ['\t'+word_substitute(expr, to_replace)]

    for name, expr in parameters.iteritems():
        struct_parameters += [expr]

    for name, expr in func_declarations.iteritems():
        func_begin += [expr]

    allocate_y_vector += ['\treturn <double*>malloc({diff_num}*sizeof(double))'.format(diff_num=len(diff_vars.keys()))]

    set_dimension += ['\tdimension[0] = {diff_num}'.format(diff_num=len(diff_vars.keys()))]

    everything = struct_parameters + set_dimension + allocate_y_vector + func_fill_yvector + func_empty_yvector + func_begin + func_end
    return everything

def add_GSL_declarations(vector_code, variables, extra_information):
    '''
    This function writes the initialization of the variables in the params struct, that are needed in func
    it does so by analyzing the variable declarations brian does already together with the vector_code to see which
    variables are needed.
    '''
    other_variables = extra_information['other_variables']
    scalar_variables = extra_information['scalar_variables']
    vector_variables = extra_information['vector_variables']
    variable_mapping = extra_information['variable_mapping']
    datatypes = find_datatypes(variables, other_variables)

    defined = ['t', 'dt', '_idx']

    expressions = []

    for var in variable_mapping:
        if var in defined:
            continue
        expressions += ['p.{ptr_name} = {array_name};'.format(ptr_name=variable_mapping[var]['pointer'],
                                                              array_name=variable_mapping[var]['actual'])]
        defined += [var]

    for var in vector_variables:
        try:
            value = variables[var]
        except KeyError:
            value = other_variables[var]
        if isinstance(value, Function):
            continue
        if isinstance(value, AuxiliaryVariable):
            continue
        if var in defined:
            continue
        expressions += ['p.{var} = {var};'.format(var=var)]

    return expressions


def temp_add_GSL_declarations(vector_variables, variables, other_variables, scalar_variables):
    '''
    This function writes the initialization of the variables in the params struct, that are needed in func
    it does so by analyzing the variable declarations brian does already together with the vector_code to see which
    variables are needed.
    '''
    variable_mapping = find_variable_mapping(variables, other_variables)
    datatypes = find_datatypes(variables, other_variables)

    defined = ['t', '_idx']

    expressions = []

    for is_vector, variable_list in zip([True, False], [vector_variables, scalar_variables]):
        for var in variable_list:
            try:
                value = variables[var]
            except KeyError:
                value = other_variables[var]
            if isinstance(value, Function):
                continue
            if isinstance(value, AuxiliaryVariable):
                continue
            if var in defined:
                continue
            if isinstance(value, ArrayVariable):
                array_name = variable_mapping[var]
                expressions += ['{struct}{var} = <{datatype} *> _buf_{array_name}.data'.
                                format(struct='p.'*is_vector,var=array_name, datatype=datatypes[var], array_name=array_name)]
            else:
                expressions += ['{struct}{var} = _namespace["{var}"]'.format(struct='p.'*is_vector,var=var)]

    return expressions

def add_GSL_scalar_code(scalar_code, extra_information):

    scalar_variables = extra_information['scalar_variables']
    vector_variables = extra_information['vector_variables']

    code = []
    for line in scalar_code:
        try:
            var, op, expr, comment = parse_statement(line)
        except ValueError:
            code += [line]
            continue
        m = re.search('([a-z|A-Z|0-9|_]+)$', var)
        actual_var = m.group(1)
        if actual_var in scalar_variables:
            code += [line]
        if actual_var in vector_variables:
            if var == 't':
                continue
            code += ['p.{var} {op} {expr} {comment}'.format(
                    var=actual_var, op=op, expr=expr, comment=comment)]

    return code


def temp_add_GSL_scalar_code(scalar_code, scalar_variables, vector_variables):

    code = []

    for line in scalar_code:
        try:
            var, op, expr, comment = parse_statement(line)
        except ValueError:
            code += [line]
            continue
        if var in scalar_variables:
            code += [line]
        if var in vector_variables:
            if var == 't': # ignore t = _array_defaultclock_t[0] if t is used func (GSL handles t)
                continue
            code += ['p.'+line]

    return code



class LazyTemplateLoader(object):
    '''
    Helper object to load templates only when they are needed.
    '''
    def __init__(self, environment, extension):
        self.env = environment
        self.extension = extension
        self._templates = {}

    def get_template(self, name):
        if name not in self._templates:
            try:
                template = CodeObjectTemplate(self.env.get_template(name+self.extension),
                                              self.env.loader.get_source(self.env,
                                                                         name+self.extension)[0])
            except TemplateNotFound:
                try:
                    # Try without extension as well (e.g. for makefiles)
                    template = CodeObjectTemplate(self.env.get_template(name),
                                                  self.env.loader.get_source(self.env,
                                                                             name)[0])
                except TemplateNotFound:
                    raise KeyError('No template with name "%s" found.' % name)
            self._templates[name] = template
        return self._templates[name]


class Templater(object):
    '''
    Class to load and return all the templates a `CodeObject` defines.

    Parameters
    ----------

    package_name : str, tuple of str
        The package where the templates are saved. If this is a tuple then each template will be searched in order
        starting from the first package in the tuple until the template is found. This allows for derived templates
        to be used. See also `~Templater.derive`.
    env_globals : dict (optional)
        A dictionary of global values accessible by the templates. Can be used for providing utility functions.
        In all cases, the filter 'autoindent' is available (see existing templates for example usage).

    Notes
    -----
    Templates are accessed using ``templater.template_base_name`` (the base name is without the file extension).
    This returns a `CodeObjectTemplate`.
    '''
    def __init__(self, package_name, extension, env_globals=None):
        if isinstance(package_name, basestring):
            package_name = (package_name,)
        loader = ChoiceLoader([PackageLoader(name, 'templates') for name in package_name])
        self.env = Environment(loader=loader, trim_blocks=True,
                               lstrip_blocks=True, undefined=StrictUndefined)
        self.env.globals['autoindent'] = autoindent
        self.env.filters['autoindent'] = autoindent
        self.env.filters['variables_to_array_names'] = variables_to_array_names
        self.env.filters['write_GSL_support_code'] = write_GSL_support_code
        self.env.filters['add_GSL_declarations'] = add_GSL_declarations
        self.env.filters['add_GSL_scalar_code'] = add_GSL_scalar_code
        if env_globals is not None:
            self.env.globals.update(env_globals)
        else:
            env_globals = {}
        self.env_globals = env_globals
        self.package_names = package_name
        self.extension = extension
        self.templates = LazyTemplateLoader(self.env, extension)

    def __getattr__(self, item):
        return self.templates.get_template(item)

    def derive(self, package_name, extension=None, env_globals=None):
        '''
        Return a new Templater derived from this one, where the new package name and globals overwrite the old.
        '''
        if extension is None:
            extension = self.extension
        if isinstance(package_name, basestring):
            package_name = (package_name,)
        if env_globals is None:
            env_globals = {}
        package_name = package_name+self.package_names
        new_env_globals = self.env_globals.copy()
        new_env_globals.update(**env_globals)
        return Templater(package_name, extension=extension,
                         env_globals=new_env_globals)


class CodeObjectTemplate(object):
    '''
    Single template object returned by `Templater` and used for final code generation

    Should not be instantiated by the user, but only directly by `Templater`.

    Notes
    -----

    The final code is obtained from this by calling the template (see `~CodeObjectTemplater.__call__`).
    '''
    def __init__(self, template, template_source):
        self.template = template
        self.template_source = template_source
        #: The set of variables in this template
        self.variables = set([])
        #: The indices over which the template iterates completely
        self.iterate_all = set([])
        #: Read-only variables that are changed by this template
        self.writes_read_only = set([])
        # This is the bit inside {} for USES_VARIABLES { list of words }
        specifier_blocks = re.findall(r'\bUSES_VARIABLES\b\s*\{(.*?)\}',
                                      template_source, re.M|re.S)
        # Same for ITERATE_ALL
        iterate_all_blocks = re.findall(r'\bITERATE_ALL\b\s*\{(.*?)\}',
                                        template_source, re.M|re.S)
        # And for WRITES_TO_READ_ONLY_VARIABLES
        writes_read_only_blocks = re.findall(r'\bWRITES_TO_READ_ONLY_VARIABLES\b\s*\{(.*?)\}',
                                             template_source, re.M|re.S)
        #: Does this template allow writing to scalar variables?
        self.allows_scalar_write = 'ALLOWS_SCALAR_WRITE' in template_source

        for block in specifier_blocks:
            self.variables.update(get_identifiers(block))
        for block in iterate_all_blocks:
            self.iterate_all.update(get_identifiers(block))
        for block in writes_read_only_blocks:
            self.writes_read_only.update(get_identifiers(block))
                
    def __call__(self, scalar_code, vector_code, **kwds):
        '''
        Return a usable code block or blocks from this template.

        Parameters
        ----------
        scalar_code : dict
            Dictionary of scalar code blocks.
        vector_code : dict
            Dictionary of vector code blocks
        **kwds
            Additional parameters to pass to the template

        Notes
        -----

        Returns either a string (if macros were not used in the template), or a `MultiTemplate` (if macros were used).
        '''
        if scalar_code is not None and len(scalar_code)==1 and scalar_code.keys()[0] is None:
            scalar_code = scalar_code[None]
        if vector_code is not None and len(vector_code)==1 and vector_code.keys()[0] is None:
            vector_code = vector_code[None]
        kwds['scalar_code'] = scalar_code
        kwds['vector_code'] = vector_code
        module = self.template.make_module(kwds)
        if len([k for k in module.__dict__.keys() if not k.startswith('_')]):
            return MultiTemplate(module)
        else:
            return autoindent_postfilter(str(module))


class MultiTemplate(Mapping):
    '''
    Code generated by a `CodeObjectTemplate` with multiple blocks

    Each block is a string stored as an attribute with the block name. The
    object can also be accessed as a dictionary.
    '''
    def __init__(self, module):
        self._templates = {}
        for k, f in module.__dict__.items():
            if not k.startswith('_'):
                s = autoindent_postfilter(str(f()))
                setattr(self, k, s)
                self._templates[k] = s

    def __getitem__(self, item):
        return self._templates[item]

    def __iter__(self):
        return iter(self._templates)

    def __len__(self):
        return len(self._templates)

    def __str__(self):
        s = ''
        for k, v in self._templates.items():
            s += k+':\n'
            s += strip_empty_lines(indent(v))+'\n'
        return s
    
    __repr__ = __str__
