"""
Package providing import/export support.
"""

from .dictlike import *
from .importexport import *

__all__ = ["ImportExport"]

# Register the standard ImportExport methods
for obj in [DictImportExport(), PandasImportExport()]:
    ImportExport.register(obj)
