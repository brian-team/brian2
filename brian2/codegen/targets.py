"""
Module that stores all known code generation targets as `codegen_targets`.
"""

__all__ = ["codegen_targets"]

# This should be filled in by subpackages
#
#: Set of all registered code generation target classes.
#: Each target is a CodeObject subclass with a `class_name` attribute.
#: Targets register themselves by calling codegen_targets.add(TargetClass)
codegen_targets = set()
