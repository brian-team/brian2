'''
Base class for generating code in different programming languages, gives the
methods which should be overridden to implement a new language.
'''
from brian2.core.variables import ArrayVariable
from brian2.core.functions import Function
from brian2.utils.stringtools import get_identifiers
from brian2.utils.logger import get_logger
from brian2.codegen.translation import make_statements
from brian2.codegen.permutation_analysis import (check_for_order_independence,
                                                 OrderDependenceError)

__all__ = ['CodeGenerator']


logger = get_logger(__name__)


class CodeGenerator(object):
    '''
    Base class for all languages.
    
    See definition of methods below.
    
    TODO: more details here
    '''

    # Subclasses should override this
    class_name = ''

    def __init__(self, variables, variable_indices, owner, iterate_all,
                 codeobj_class, name, template_name,
                 override_conditional_write=None,
                 allows_scalar_write=False):
        # We have to do the import here to avoid circular import dependencies.
        from brian2.devices.device import get_device
        self.device = get_device()
        self.variables = variables
        self.variable_indices = variable_indices
        self.func_name_replacements = {}
        for varname, var in variables.iteritems():
            if isinstance(var, Function):
                if codeobj_class in var.implementations:
                    impl_name = var.implementations[codeobj_class].name
                    if impl_name is not None:
                        self.func_name_replacements[varname] = impl_name
        self.iterate_all = iterate_all
        self.codeobj_class = codeobj_class
        self.owner = owner
        if override_conditional_write is None:
            self.override_conditional_write = set()
        else:
            self.override_conditional_write = set(override_conditional_write)
        self.allows_scalar_write = allows_scalar_write
        self.name = name
        self.template_name = template_name

    @staticmethod
    def get_array_name(var, access_data=True):
        '''
        Get a globally unique name for a `ArrayVariable`.

        Parameters
        ----------
        var : `ArrayVariable`
            The variable for which a name should be found.
        access_data : bool, optional
            For `DynamicArrayVariable` objects, specifying `True` here means the
            name for the underlying data is returned. If specifying `False`,
            the name of object itself is returned (e.g. to allow resizing).
        Returns
        -------
        name : str
            A uniqe name for `var`.
        '''
        # We have to do the import here to avoid circular import dependencies.
        from brian2.devices.device import get_device
        device = get_device()
        return device.get_array_name(var, access_data=access_data)

    def translate_expression(self, expr):
        '''
        Translate the given expression string into a string in the target
        language, returns a string.
        '''
        raise NotImplementedError

    def translate_statement(self, statement):
        '''
        Translate a single line `Statement` into the target language, returns
        a string.
        '''
        raise NotImplementedError

    def determine_keywords(self):
        '''
        A dictionary of values that is made available to the templated. This is
        used for example by the `CPPCodeGenerator` to set up all the supporting
        code
        '''
        raise NotImplementedError

    def translate_one_statement_sequence(self, statements, scalar=False):
        raise NotImplementedError

    def translate_statement_sequence(self, scalar_statements, vector_statements):
        '''
        Translate a sequence of `Statement` into the target language, taking
        care to declare variables, etc. if necessary.
   
        Returns a tuple ``(scalar_code, vector_code, kwds)`` where
        ``scalar_code`` is a list of the lines of code executed before the inner
        loop, ``vector_code`` is a list of the lines of code in the inner
        loop, and ``kwds`` is a dictionary of values that is made available to
        the template.
        '''
        scalar_code = {}
        vector_code = {}
        for name, block in scalar_statements.iteritems():
            scalar_code[name] = self.translate_one_statement_sequence(block, scalar=True)
        for name, block in vector_statements.iteritems():
            vector_code[name] = self.translate_one_statement_sequence(block, scalar=False)

        kwds = self.determine_keywords()

        return scalar_code, vector_code, kwds

    def array_read_write(self, statements):
        '''
        Helper function, gives the set of ArrayVariables that are read from and
        written to in the series of statements. Returns the pair read, write
        of sets of variable names.
        '''
        variables = self.variables
        variable_indices = self.variable_indices
        read = set()
        write = set()
        for stmt in statements:
            ids = get_identifiers(stmt.expr)
            # if the operation is inplace this counts as a read.
            if stmt.inplace:
                ids.add(stmt.var)
            read = read.union(ids)
            if stmt.scalar or variable_indices[stmt.var] == '0':
                if stmt.op != ':=' and not self.allows_scalar_write:
                    raise SyntaxError(('Writing to scalar variable %s '
                                       'not allowed in this context.' % stmt.var))
                for name in ids:
                    if (name in variables and isinstance(variables[name], ArrayVariable)
                                          and not (variables[name].scalar or
                                                           variable_indices[name] == '0')):
                        raise SyntaxError(('Cannot write to scalar variable %s '
                                           'with an expression referring to '
                                           'vector variable %s') %
                                          (stmt.var, name))
            write.add(stmt.var)
        read = set(varname for varname, var in variables.items()
                   if isinstance(var, ArrayVariable) and varname in read)
        write = set(varname for varname, var in variables.items()
                    if isinstance(var, ArrayVariable) and varname in write)
        # Gather the indices stored as arrays (ignore _idx which is special)
        indices = set()
        indices |= set(variable_indices[varname] for varname in read
                       if not variable_indices[varname] in ('_idx', '0')
                           and isinstance(variables[variable_indices[varname]],
                                          ArrayVariable))
        indices |= set(variable_indices[varname] for varname in write
                       if not variable_indices[varname] in ('_idx', '0')
                           and isinstance(variables[variable_indices[varname]],
                                          ArrayVariable))
        # don't list arrays that are read explicitly and used as indices twice
        read -= indices
        return read, write, indices

    def get_conditional_write_vars(self):
        '''
        Helper function, returns a dict of mappings ``(varname, condition_var_name)`` indicating that
        when ``varname`` is written to, it should only be when ``condition_var_name`` is ``True``.
        '''
        conditional_write_vars = {}
        for varname, var in self.variables.items():
            if getattr(var, 'conditional_write', None) is not None:
                cvar = var.conditional_write
                cname = cvar.name
                if cname not in self.override_conditional_write:
                    conditional_write_vars[varname] = cname
        return conditional_write_vars

    def arrays_helper(self, statements):
        '''
        Combines the two helper functions `array_read_write` and `get_conditional_write_vars`, and updates the
        ``read`` set.
        '''
        read, write, indices = self.array_read_write(statements)
        conditional_write_vars = self.get_conditional_write_vars()
        read |= set(var for var in write
                    if var in conditional_write_vars)
        read |= set(conditional_write_vars[var] for var in write
                    if var in conditional_write_vars)
        return read, write, indices, conditional_write_vars

    def has_repeated_indices(self, statements):
        '''
        Whether any of the statements potentially uses repeated indices (e.g.
        pre- or postsynaptic indices).
        '''
        variables = self.variables
        variable_indices = self.variable_indices
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        # Check whether we potentially deal with repeated indices (which will
        # be the case most importantly when we write to pre- or post-synaptic
        # variables in synaptic code)
        used_indices = set(variable_indices[var] for var in write)
        all_unique = all(variables[index].unique for index in used_indices
                         if index not in ('_idx', '0'))
        return not all_unique

    def translate(self, code, dtype):
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
                                                 self.variables,
                                                 self.variable_indices)
            except OrderDependenceError:
                # If the abstract code is only one line, display it in full
                if len(vs) <= 1:
                    error_msg = 'Abstract code: "%s"\n' % vs[0]
                else:
                    error_msg = ('%d lines of abstract code, first line is: '
                                 '"%s"\n') % (len(vs), vs[0])
                logger.warn(('Came across an abstract code block that may not be '
                             'well-defined: the outcome may depend on the '
                             'order of execution. You can ignore this warning if '
                             'you are sure that the order of operations does not '
                             'matter. ' + error_msg))

        translated = self.translate_statement_sequence(scalar_statements,
                                                       vector_statements)

        return translated
