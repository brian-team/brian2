"""
cppyy Runtime Module
====================
This module provides JIT C++ compilation for Brian2 using cppyy.
"""

from .cppyy_rt import CppyyCodeObject

__all__ = ["CppyyCodeObject"]
