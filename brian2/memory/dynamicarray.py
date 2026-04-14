try:
    from .cythondynamicarray import DynamicArray, DynamicArray1D
except ImportError as e:
    raise ImportError(
        "DynamicArray is now compiled from Cython. Please ensure the extension is built.\n"
        "If you're running from source, try: pip install -e ."
    ) from e

__all__ = ["DynamicArray", "DynamicArray1D"]
