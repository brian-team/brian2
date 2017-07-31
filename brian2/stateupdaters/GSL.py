from .base import StateUpdateMethod, UnsupportedEquationsException
from ..core.preferences import prefs
from ..devices.device import auto_target

__all__ = ['GSL_stateupdater']

class GSLStateUpdater(StateUpdateMethod):
    '''
    A statupdater that rewrites the differential equations so that the GSL generator knows how to write the
    code in the target language.
    '''
    def get_codeobj_class(self):
        '''
        Return codeobject class based on target language and device.

        Choose which version of the GSL `CodeObject` to use. If ```isinstance(device, CPPStandaloneDevice)```, then
        we want the `GSLCPPStandaloneCodeObject`. Otherwise the return value is based on prefs.codegen.target.

        Returns
        -------
        CodeObject : `GSLWeaveCodeObject`, `GSLCythonCodeObject` or `GSLCPPStandaloneCodeObject`.
        '''
        # imports in this function to avoid circular imports
        from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
        from brian2.devices.device import get_device
        from ..codegen.runtime.GSLweave_rt import GSLWeaveCodeObject
        from ..codegen.runtime.GSLcython_rt import GSLCythonCodeObject

        device = get_device()
        if isinstance(device, CPPStandaloneDevice):
            from ..devices.cpp_standalone.GSLcodeobject import GSLCPPStandaloneCodeObject
            return GSLCPPStandaloneCodeObject

        if prefs.codegen.target == 'auto':
            target_name = auto_target().class_name
        else:
            target_name = prefs.codegen.target

        if target_name == 'cython':
            return GSLCythonCodeObject
        elif target_name == 'weave':
            return GSLWeaveCodeObject
        else:
            raise NotImplementedError

    def transfer_codeobj_class(self, obj):
        '''
        Transfer the code object class saved in self to the object sent as an argument.

        This method is returned when calling `GSLStateUpdater`. This class inherits from `StateUpdateMethod` which
        orignally only returns abstract code. However, with GSL this returns a method because more is needed than just
        the abstract code: the state updater requires its own CodeObject that is different from the other `NeuronGroup`
        objects. This method adds this `CodeObject` to the `StateUpdater` object (and also adds the variables 't', 'dt',
        and other variables that are needed in the `GSLCodeGenerator`.

        Parameters
        ----------
        obj : `GSLStateUpdater`
            the object that the codeobj_class and other variables need to be transferred to

        Returns
        -------
        str
            The abstract code (translated equations), that is returned conventionally by brian and used for later
            code generation in the `CodeGenerator.translate` method.
        '''
        obj.codeobj_class = self.get_codeobj_class()
        obj.codeobj_class.variable_flags = self.flags #TODO: temporary solution for sending flags to generator
        obj.needed_variables += ['t', 'dt'] + self.needed_variables
        return GSL_stateupdater.abstract_code

    def __call__(self, equations, variables=None):
        '''
        Translate equations to abstract_code.

        Parameters
        ----------
        equations : `Equations`
            object containing the equations that describe the ODE system
        variables : dict
            dictionary containing str, `Variable` pairs

        Returns
        -------
        method
            Method that needs to be called with `StateUpdater` to add CodeObject class and some other variables so
            these can be sent to the `CodeGenerator`
        '''
        if equations.is_stochastic:
            raise UnsupportedEquationsException('Cannot solve stochastic '
                                                'equations with this state '
                                                'updater.')

        # the approach is to 'tag' the differential equation variables so they can
        # be translated to GSL code
        diff_eqs = equations.get_substituted_expressions(variables)

        code = []
        count_statevariables = 0
        counter = {}
        self.needed_variables = []

        for diff_name, expr in diff_eqs:
            # if diff_name does not occur in the right hand side of the equation, Brian does not
            # know to add the variable to the namespace, so we add it to needed_variables
            self.needed_variables += [diff_name]
            counter[diff_name] = count_statevariables
            code += ['_gsl_{var}_f{count} = {expr}'.format(var=diff_name,
                                                               expr=expr,
                                                               count=counter[diff_name])]
            count_statevariables += 1

        # add flags to variables objects because some of them we need in the GSL generator
        flags = {}
        for eq_name, eq_obj in equations._equations.items():
            if len(eq_obj.flags) > 0:
                flags[eq_name] = eq_obj.flags

        self.abstract_code =  ('\n').join(code)
        self.flags = flags #TODO: temporary solution for sending flags to generator
        return self.transfer_codeobj_class

    # Copy doc from parent class
    __call__.__doc__ = StateUpdateMethod.__call__.__doc__

GSL_stateupdater = GSLStateUpdater()
