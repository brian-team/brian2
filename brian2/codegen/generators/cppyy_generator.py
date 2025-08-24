"""
cppyy Code Generator
====================
This generator converts Brian2's abstract code into C++ code that can be
JIT-compiled by cppyy. It inherits from CPPCodeGenerator to reuse most
C++ generation logic.

WHY: We need this to translate Brian's abstract syntax tree (AST) into C++ code.
"""

import typing

from brian2.codegen.generators.cpp_generator import CPPCodeGenerator
from brian2.core.preferences import BrianPreference, prefs

# Register cppyy-specific preferences
prefs.register_preferences(
    "codegen.generators.cppyy",
    "cppyy codegen preferences",
    restrict_keyword=BrianPreference(
        default="",  # No restrict keyword for cppyy
        docs="""
         The restrict keyword for cppyy. Empty by default as cppyy
         doesn't always handle __restrict well.
         """,
    ),
    flush_denormals=BrianPreference(
        default=False,
        docs="""
         Whether to add denormal flushing code. Disabled for cppyy
         as it's handled at runtime.
         """,
    ),
)


class CppyyCodeGenerator(CPPCodeGenerator):
    """
    cppyy code generator - generates C++ code for JIT compilation

    This class handles the conversion of Brian2's abstract code representation
    into C++ code that cppyy can compile and execute.
    """

    class_name = "cppyy"

    def __init__(self, *args, **kwds):
        """
        Initialize the cppyy code generator.

        WHY: We call the parent class (CPPCodeGenerator) because we want to
        reuse majority of its functionality. We're not reinventing C++ generation,
        just adapting it for cppyy.
        """
        super().__init__(*args, **kwds)

        # # Track what headers we need for cppyy
        # # WHY: cppyy needs to know which C++ headers to include
        # self.headers_to_include = [
        #     "<cmath>",  # For mathematical functions
        #     "<algorithm>",  # For min, max, etc.
        #     "<cstdint>",  # For int32_t, int64_t types
        # ]

    @property
    def restrict(self):
        """Override to use cppyy-specific preference"""
        # Return empty string or use cppyy preference
        return f"{prefs['codegen.generators.cppyy.restrict_keyword']}"

    @property
    def flush_denormals(self):
        """Override to use cppyy-specific preference."""
        return prefs["codegen.generators.cppyy.flush_denormals"]

    def translate_expression(self, expr):
        """
        Translate a Brian2 expression to C++ code.

        Example:
            Input: "v + 1"
            Output: "_ptr_array_neurongroup_v[_idx] + 1"

        WHY: Brian2 uses simple variable names (v), but in C++ we need to
        access them as array elements with proper indexing.
        """
        # Use parent class method - it already handles this well
        return super().translate_expression(expr)

    def determine_keywords(self) -> dict[str, typing.Any]:
        """
        Determine which C++ keywords are used in the generated code.

        WHY: This helps optimize the generated code by only including
        necessary type definitions and functions.
        """
        # Get all standard CPP keywords
        keywords: dict[str, typing.Any] = (
            super().determine_keywords()
        )  # satisfy Pyright

        keywords.update(
            {
                "is_cppyy_target": True,
                "is_standalone": False,
                "cppyy_function_name": f"brian_kernel_{self.name}",
                # These help templates know they're in runtime mode
                "runtime_mode": True,
                "needs_main_function": False,
            }
        )

        # Modify support code for cppyy
        # We don't need file I/O or main function setup
        keywords["support_code_lines"] = self._adapt_support_code(
            keywords.get("support_code_lines", "")
        )
        return keywords

    def _adapt_support_code(self, support_code):
        """
        Adapt support code for cppyy runtime.
        Remove file I/O, adapt for JIT compilation.
        """
        # TODO: For cppyy, we compile support code separately, so we might
        # want to split it into header-like and implementation parts
        return support_code
