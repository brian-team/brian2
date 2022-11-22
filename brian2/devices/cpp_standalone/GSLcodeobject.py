"""
Module containing CPPStandalone CodeObject for code generation for integration using the ODE solver provided in the
GNU Scientific Library
"""

from brian2.codegen.codeobject import CodeObject
from brian2.codegen.generators.cpp_generator import CPPCodeGenerator
from brian2.codegen.generators.GSL_generator import GSLCPPCodeGenerator
from brian2.devices.cpp_standalone import CPPStandaloneCodeObject


class GSLCPPStandaloneCodeObject(CodeObject):
    templater = CPPStandaloneCodeObject.templater.derive(
        "brian2.devices.cpp_standalone", templates_dir="templates_GSL"
    )
    original_generator_class = CPPCodeGenerator
    generator_class = GSLCPPCodeGenerator
