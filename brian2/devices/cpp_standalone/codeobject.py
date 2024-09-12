"""
Module implementing the C++ "standalone" `CodeObject`
"""

from brian2.codegen.codeobject import CodeObject, constant_or_scalar
from brian2.codegen.generators.cpp_generator import CPPCodeGenerator, c_data_type
from brian2.codegen.targets import codegen_targets
from brian2.codegen.templates import Templater
from brian2.core.functions import DEFAULT_FUNCTIONS
from brian2.core.preferences import prefs
from brian2.devices.device import get_device
from brian2.utils.stringtools import replace

__all__ = ["CPPStandaloneCodeObject"]


def openmp_pragma(pragma_type):
    nb_threads = prefs.devices.cpp_standalone.openmp_threads
    openmp_on = not (nb_threads == 0)

    ## First we need to deal with some special cases that have to be handle in case
    ## openmp is not activated
    if pragma_type == "set_num_threads":
        if not openmp_on:
            return ""
        elif nb_threads > 0:
            # We have to fix the exact number of threads in all parallel sections
            return f"omp_set_dynamic(0);\nomp_set_num_threads({int(nb_threads)});"
    elif pragma_type == "get_thread_num":
        if not openmp_on:
            return "0"
        else:
            return "omp_get_thread_num()"
    elif pragma_type == "get_num_threads":
        if not openmp_on:
            return "1"
        else:
            return f"{int(nb_threads)}"
    elif pragma_type == "with_openmp":
        # The returned value is a proper Python boolean, i.e. not something
        # that should be included in the generated code but rather for use
        # in {% if ... %} statements in the template
        return openmp_on

    ## Then by default, if openmp is off, we do not return any pragma statement in the templates
    elif not openmp_on:
        return ""
    ## Otherwise, we return the correct pragma statement
    elif pragma_type == "static":
        return "#pragma omp for schedule(static)"
    elif pragma_type == "single":
        return "#pragma omp single"
    elif pragma_type == "single-nowait":
        return "#pragma omp single nowait"
    elif pragma_type == "critical":
        return "#pragma omp critical"
    elif pragma_type == "atomic":
        return "#pragma omp atomic"
    elif pragma_type == "once":
        return "#pragma once"
    elif pragma_type == "parallel-static":
        return "#pragma omp parallel for schedule(static)"
    elif pragma_type == "static-ordered":
        return "#pragma omp for schedule(static) ordered"
    elif pragma_type == "ordered":
        return "#pragma omp ordered"
    elif pragma_type == "include":
        return "#include <omp.h>"
    elif pragma_type == "parallel":
        return "#pragma omp parallel"
    elif pragma_type == "master":
        return "#pragma omp master"
    elif pragma_type == "barrier":
        return "#pragma omp barrier"
    elif pragma_type == "compilation":
        return "-fopenmp"
    elif pragma_type == "sections":
        return "#pragma omp sections"
    elif pragma_type == "section":
        return "#pragma omp section"
    else:
        raise ValueError(f'Unknown OpenMP pragma "{pragma_type}"')


class CPPStandaloneCodeObject(CodeObject):
    """
    C++ standalone code object

    The ``code`` should be a `~brian2.codegen.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    """

    templater = Templater(
        "brian2.devices.cpp_standalone",
        ".cpp",
        env_globals={
            "c_data_type": c_data_type,
            "openmp_pragma": openmp_pragma,
            "constant_or_scalar": constant_or_scalar,
            "prefs": prefs,
            "zip": zip,
        },
    )
    generator_class = CPPCodeGenerator

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        #: Store whether this code object defines before/after blocks
        self.before_after_blocks = []

    def __call__(self, **kwds):
        return self.run()

    def compile_block(self, block):
        pass  # Compilation will be handled in device

    def run_block(self, block):
        if block == "run":
            get_device().main_queue.append((f"{block}_code_object", (self,)))
        else:
            # Check the C++ code whether there is anything to run
            cpp_code = getattr(self.code, f"{block}_cpp_file")
            if len(cpp_code) and "EMPTY_CODE_BLOCK" not in cpp_code:
                get_device().main_queue.append((f"{block}_code_object", (self,)))
                self.before_after_blocks.append(block)


codegen_targets.add(CPPStandaloneCodeObject)


# At module initialization time, we do not yet know whether the code will be
# run with OpenMP or not. We therefore use a "dynamic implementation" which
# generates the rand/randn implementation during code generation.
def generate_rand_code(rand_func, owner):
    nb_threads = prefs.devices.cpp_standalone.openmp_threads
    if nb_threads == 0:  # no OpenMP
        thread_number = "0"
    else:
        thread_number = "omp_get_thread_num()"
    if rand_func not in ["rand", "randn"]:
        raise AssertionError(rand_func)
    code = """
           inline double _%RAND_FUNC%(const int _vectorisation_idx) {
               return brian::_random_generators[%THREAD_NUMBER%].%RAND_FUNC%();
           }
           """
    code = replace(
        code,
        {
            "%THREAD_NUMBER%": thread_number,
            "%RAND_FUNC%": rand_func,
        },
    )
    return {"support_code": code}


rand_impls = DEFAULT_FUNCTIONS["rand"].implementations
rand_impls.add_dynamic_implementation(
    CPPStandaloneCodeObject,
    code=lambda owner: generate_rand_code("rand", owner),
    namespace=lambda owner: {},
    name="_rand",
)

randn_impls = DEFAULT_FUNCTIONS["randn"].implementations
randn_impls.add_dynamic_implementation(
    CPPStandaloneCodeObject,
    code=lambda owner: generate_rand_code("randn", owner),
    namespace=lambda owner: {},
    name="_randn",
)
