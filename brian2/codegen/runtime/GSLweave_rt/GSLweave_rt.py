from brian2.core.functions import DEFAULT_FUNCTIONS
from ..weave_rt import WeaveCodeObject
from ...generators.GSL_generator import GSLWeaveCodeGenerator
from ..weave_rt import WeaveCodeGenerator

__all__ = ['GSLWeaveCodeObject']

class GSLWeaveCodeObject(WeaveCodeObject):

    templater = WeaveCodeObject.templater.derive('brian2.codegen.runtime.GSLweave_rt')

    original_generator_class = WeaveCodeGenerator
    generator_class = GSLWeaveCodeGenerator

# Although searching for implementations with [WeaveCodeObject] gives the right code,
# it doesn't work properly when searching with GSLWeaveCodeObject (because it doesn't
# inherit anywhere from CPPCodeGenerator I believe). So it's done manually here..
for function in DEFAULT_FUNCTIONS.values():
    function.implementations._implementations[GSLWeaveCodeObject] = \
        function.implementations[WeaveCodeObject]

