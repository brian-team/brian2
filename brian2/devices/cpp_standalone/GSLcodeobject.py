'''
Module containing CPPStandalone CodeObject for code generation for integration using the ODE solver provided in the
GNU Scientific Library
'''
from brian2.codegen.codeobject import CodeObject
from brian2.devices.cpp_standalone import CPPStandaloneCodeObject
from brian2.codegen.generators.cpp_generator import CPPCodeGenerator
from brian2.codegen.generators import GSLWeaveCodeGenerator

class GSLCPPStandaloneCodeObject(CodeObject):

    templater = CPPStandaloneCodeObject.templater.derive('brian2.codegen.runtime.GSLweave_rt')
    original_generator_class = CPPCodeGenerator
    generator_class = GSLWeaveCodeGenerator
