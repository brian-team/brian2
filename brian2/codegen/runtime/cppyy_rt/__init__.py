"""
cppyy Runtime Backend for Brian2.
"""

from __future__ import annotations

from brian2.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from brian2.codegen.runtime.cppyy_rt.cppyy_rt import CppyyCodeObject
    from brian2.codegen.targets import codegen_targets

    # Register the target (same pattern as numpy_rt and cython_rt)
    codegen_targets.add(CppyyCodeObject)

    __all__ = ["CppyyCodeObject"]
    logger.debug("cppyy runtime backend registered")

except ImportError as e:
    logger.debug(f"cppyy runtime backend not available: {e}")
    __all__ = []
    CppyyCodeObject = None
