"""
Package providing import/export support.
"""

from .importexport import *
from .dictlike import *


__all__ = ["ImportExport"]

# Register the standard ImportExport methods
for obj in [DictImportExport(), PandasImportExport()]:
    ImportExport.register(obj)
