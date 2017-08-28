'''
Module containg the StateUpdateMethod for integration using the ODE solver
provided in the GNU Scientific Library (GSL)
'''
from .base import StateUpdateMethod, UnsupportedEquationsException
from ..core.preferences import prefs
from ..devices.device import auto_target
from brian2.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ['gsl_rk2', 'gsl_rk4', 'gsl_rkf45', 'gsl_rkck', 'gsl_rk8pd']

class GSLContainer(object):
    '''
    Class that contains information (equation- or integrator-related) required
    for later code generation
    '''
    def __init__(self, integrator, abstract_code=None, needed_variables=[], variable_flags=[]):
        self.integrator = integrator
        self.abstract_code = abstract_code
        self.needed_variables = needed_variables
        self.variable_flags = variable_flags

    def get_codeobj_class(self):
        '''
        Return codeobject class based on target language and device.

        Choose which version of the GSL `CodeObject` to use. If
        ```isinstance(device, CPPStandaloneDevice)```, then
        we want the `GSLCPPStandaloneCodeObject`. Otherwise the return value is
        based on prefs.codegen.target.

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
            raise NotImplementedError(("Selected stateupdater is GSL stateupdater, while "
                                       "target language is numpy (either selected or only "
                                       "one available): GSL was not implemented for numpy."
                                       "\nSet target language to weave or cython, or device "
                                       "to cpp_standalone in order to use GSL integration"))

    def __call__(self, obj):
        '''
        Transfer the code object class saved in self to the object sent as an argument.

        This method is returned when calling `GSLStateUpdater`. This class inherits
        from `StateUpdateMethod` which orignally only returns abstract code.
        However, with GSL this returns a method because more is needed than just
        the abstract code: the state updater requires its own CodeObject that is
        different from the other `NeuronGroup` objects. This method adds this
        `CodeObject` to the `StateUpdater` object (and also adds the variables
        't', 'dt', and other variables that are needed in the `GSLCodeGenerator`.

        Parameters
        ----------
        obj : `GSLStateUpdater`
            the object that the codeobj_class and other variables need to be transferred to

        Returns
        -------
        str
            The abstract code (translated equations), that is returned conventionally
            by brian and used for later code generation in the `CodeGenerator.translate` method.
        '''
        obj.codeobj_class = self.get_codeobj_class()
        obj._gsl_variable_flags = self.variable_flags
        if getattr(obj, 'method_options', None) is None:
            obj.method_options = {}
        obj.method_options['integrator'] = self.integrator
        obj.needed_variables += ['t', 'dt'] + self.needed_variables
        return self.abstract_code

class GSLStateUpdater(StateUpdateMethod):
    '''
    A statupdater that rewrites the differential equations so that the GSL generator
    knows how to write the code in the target language.
    '''
    def __init__(self, integrator):
        self.integrator = integrator

    def __call__(self, equations, variables=None):
        '''
        Translate equations to abstract_code.

        Parameters
        ----------a
        equations : `Equations`
            object containing the equations that describe the ODE systemTransferClass(self)
        variables : dict
            dictionary containing str, `Variable` pairs

        Returns
        -------
        method
            Method that needs to be called with `StateUpdater` to add CodeObject
            class and some other variables so these can be sent to the `CodeGenerator`
        '''
        logger.warn("Integrating equations with GSL is still considered experimental", once=True)

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
        diff_vars = []

        for diff_name, expr in diff_eqs:
            # if diff_name does not occur in the right hand side of the equation, Brian does not
            # know to add the variable to the namespace, so we add it to needed_variables
            diff_vars += [diff_name]
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

        return GSLContainer(self.integrator,
                            abstract_code=('\n').join(code),
                            needed_variables=diff_vars,
                            variable_flags=flags)

    # Copy doc from parent class
    __call__.__doc__ = StateUpdateMethod.__call__.__doc__

gsl_rk2 = GSLStateUpdater('rk2')
gsl_rk4 = GSLStateUpdater('rk4')
gsl_rkf45 = GSLStateUpdater('rkf45')
gsl_rkck = GSLStateUpdater('rkck')
gsl_rk8pd = GSLStateUpdater('rk8pd')
