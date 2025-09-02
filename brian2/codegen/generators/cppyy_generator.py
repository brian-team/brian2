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

    def translate(self, abstract_code, dtype):
        """Override to flatten the generated code structure for cppyy templates"""

        # Get the standard CPP generator result
        scalar_code, vector_code, kwds = super().translate(abstract_code, dtype)

        print("\n=== DEBUGGING CODE TRANSLATION (Before Flattening) ===")
        print(f"Raw scalar_code type: {type(scalar_code)}")
        print(f"Raw scalar_code: {scalar_code}")
        print(f"Raw vector_code type: {type(vector_code)}")
        print(f"Raw vector_code: {vector_code}")

        # Flatten the code structures into simple strings
        flattened_scalar = self._flatten_code_block(scalar_code)
        flattened_vector = self._flatten_code_block(vector_code)

        print(f"Flattened scalar_code: '{flattened_scalar}'")
        print(f"Flattened vector_code: '{flattened_vector}'")
        print("=" * 60)

        return flattened_scalar, flattened_vector, kwds

    def _flatten_code_block(self, code_block):
        """
        Convert Brian2's multi-block code structure into a simple string.

        This handles the conversion from:
          {None: ['line1', 'line2', 'line3']}
        To:
          "line1\nline2\nline3"
        """

        if isinstance(code_block, str):
            # Already a simple string, return as-is
            return code_block

        if isinstance(code_block, dict):
            # This is the multi-block structure we need to flatten
            all_lines = []

            # Process each block (usually just None for simple cases)
            for _, line_list in code_block.items():
                if isinstance(line_list, list):
                    # Join all lines in this block
                    for line in line_list:
                        if line.strip():  # Skip empty lines
                            all_lines.append(line)
                elif isinstance(line_list, str):
                    # Sometimes it's already a string
                    if line_list.strip():
                        all_lines.append(line_list)

            # Join all lines with newlines to create proper C++ code
            return "\n".join(all_lines)

        if isinstance(code_block, list):
            # Sometimes it's just a list directly
            return "\n".join(line for line in code_block if line.strip())

        # Fallback: convert to string
        return str(code_block)

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
        print(f"Final keywords: {list(keywords.keys())}")
        print("=" * 50)
        return keywords

    def _adapt_support_code(self, support_code):
        """
        Adapt support code for cppyy runtime.
        Remove file I/O, adapt for JIT compilation.
        """
        # TODO: For cppyy, we compile support code separately, so we might
        # want to split it into header-like and implementation parts
        return support_code
