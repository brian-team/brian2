"""
Module to support caching of function results to memory (used to cache results
of parsing, generation of state update code, etc.). Provides the `cached`
decorator.
"""

import functools
import collections
from collections.abc import Mapping


class CacheKey(object):
    """
    Mixin class for objects that will be used as keys for caching (e.g.
    `Variable` objects) and have to define a certain "identity" with respect
    to caching. This "identity" is different from standard Python hashing and
    equality checking: a `Variable` for example would be considered "identical"
    for caching purposes regardless which object (e.g. `NeuronGroup`) it belongs
    to (because this does not matter for parsing, creating abstract code, etc.)
    but this of course matters for the values it refers to and therefore for
    comparison of equality to other variables.

    Classes that mix in the `CacheKey` class should re-define the
    ``_cache_irrelevant_attributes`` attribute to note all the attributes that
    should be ignored. The property ``_state_tuple`` will refer to a tuple of
    all attributes that were not excluded in such a way; this tuple will be used
    as the key for caching purposes.
    """

    #: Set of attributes that should not be considered for caching of state
    #: update code, etc.
    _cache_irrelevant_attributes = set()

    @property
    def _state_tuple(self):
        """A tuple with this object's attribute values, defining its identity
        for caching purposes. See `CacheKey` for details."""
        return tuple(
            value
            for key, value in sorted(self.__dict__.items())
            if key not in self._cache_irrelevant_attributes
        )


class _CacheStatistics(object):
    """
    Helper class to store cache statistics
    """

    def __init__(self):
        self.hits = 0
        self.misses = 0

    def __repr__(self):
        return f"<Cache statistics: {int(self.hits)} hits, {int(self.misses)} misses>"


def cached(func):
    """
    Decorator to cache a function so that it will not be re-evaluated when
    called with the same arguments. Uses the `_hashable` function to make
    arguments usable as a dictionary key even though they mutable (lists,
    dictionaries, etc.).

    Notes
    -----
    This is *not* a general-purpose caching decorator in any way comparable to
    ``functools.lru_cache`` or joblib's caching functions. It is very simplistic
    (no maximum cache size, no normalization of calls, e.g. ``foo(3)`` and
    ``foo(x=3)`` are not considered equivalent function calls) and makes very
    specific assumptions for our use case. Most importantly, `Variable` objects
    are considered to be identical when they refer to the same object, even
    though the actually stored values might have changed.

    Parameters
    ----------
    func : function
        The function to decorate.

    Returns
    -------
    decorated : function
        The decorated function.
    """
    # For simplicity, we store the cache in the function itself
    func._cache = {}
    func._cache_statistics = _CacheStatistics()

    @functools.wraps(func)
    def cached_func(*args, **kwds):
        try:
            cache_key = tuple(
                [_hashable(arg) for arg in args]
                + [(key, _hashable(value)) for key, value in sorted(kwds.items())]
            )
        except TypeError:
            # If we cannot handle a type here, that most likely means that the
            # user provided an argument of a type we don't handle. This will
            # lead to an error message later that is most likely more meaningful
            # to the user than an error message by the caching system
            # complaining about an unsupported type.
            return func(*args, **kwds)
        if cache_key in func._cache:
            func._cache_statistics.hits += 1
        else:
            func._cache_statistics.misses += 1
            func._cache[cache_key] = func(*args, **kwds)
        return func._cache[cache_key]

    return cached_func


_of_type_cache = collections.defaultdict(set)


def _of_type(obj_type, check_type):
    if (obj_type, check_type) not in _of_type_cache:
        _of_type_cache[(obj_type, check_type)] = issubclass(obj_type, check_type)
    return _of_type_cache[(obj_type, check_type)]


def _hashable(obj):
    """Helper function to make a few data structures hashable (e.g. a
    dictionary gets converted to a frozenset). The function is specifically
    tailored to our use case and not meant to be generally useful."""
    if hasattr(obj, "_state_tuple"):
        return _hashable(obj._state_tuple)
    obj_type = type(obj)
    if _of_type(obj_type, Mapping):
        return frozenset(
            (_hashable(key), _hashable(value)) for key, value in obj.items()
        )
    elif _of_type(obj_type, set):
        return frozenset(_hashable(el) for el in obj)
    elif _of_type(obj_type, tuple) or _of_type(obj_type, list):
        return tuple(_hashable(el) for el in obj)
    if hasattr(obj, "dim") and getattr(obj, "shape", None) == ():
        # Scalar Quantity object
        return float(obj), obj.dim
    else:
        try:
            # Make sure that the object is hashable
            hash(obj)
            return obj
        except TypeError:
            raise TypeError(f"Do not know how to handle object of type {type(obj)}")
