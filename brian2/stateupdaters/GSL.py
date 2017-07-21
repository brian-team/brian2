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
        This function chooses which version of the GSLCodeObject is needed based on the target language and device
        (so it chooses cython weave or cpp_standalone codeobject)
        :return: GSL{target}CodeObject
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
        The method returned when calling the StateUpdateMethod (before this class only returned abstract code)
        With GSL this returns a method because more is needed than just the abstract code: the state updater requires
        its own CodeObject that is different from the other NeuronGroup objects. This method adds this CodeObject to
        the StateUpdater object (and also adds the variables 't' and 'dt' that are needed in the GSL templates!)
        :param obj: StateUpdater object
        :return: abstract code (what the StateUpdateMethod always returned in the past)
        '''
        obj.codeobj_class = self.get_codeobj_class()
        obj.needed_variables += ['t', 'dt']
        return GSL_stateupdater.abstract_code

    def __call__(self, equations, variables=None):

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

        for diff_name, expr in diff_eqs:
            counter[diff_name] = count_statevariables
            code += ['_gsl_{var}_f{count} = {expr}'.format(var=diff_name,
                                                               expr=expr,
                                                               count=counter[diff_name])]
            count_statevariables += 1


        self.abstract_code =  ('\n').join(code)
        return self.transfer_codeobj_class


    # Copy doc from parent class
    __call__.__doc__ = StateUpdateMethod.__call__.__doc__

GSL_stateupdater = GSLStateUpdater()
