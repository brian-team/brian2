try:
    from .cythonspikequeue import SpikeQueue
except ImportError as e:
    raise ImportError(
        "SpikeQueue is now compiled from Cython. Please ensure the extension is built.\n"
        "If you're running from source, try: pip install -e ."
    ) from e

__all__ = ["SpikeQueue"]
