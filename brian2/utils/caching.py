'''
Module to support caching of function results to memory (used to cache results
of parsing, generation of state update code, etc.). Provides the `cached`
decorator.
'''

import functools
import collections


class _CacheStatistics(object):
    '''
    Helper class to store cache statistics
    '''
    def __init__(self):
        self.hits = 0
        self.misses = 0

    def __repr__(self):
        return '<Cache statistics: %d hits, %d misses>' % (self.hits, self.misses)


def cached(func):
    '''
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
    '''
    # For simplicity, we store the cache in the function itself
    func._cache = {}
    func._cache_statistics = _CacheStatistics()

    @functools.wraps(func)
    def cached_func(*args, **kwds):
        cache_key = tuple([_hashable(arg) for arg in args] +
                          [(key, _hashable(value))
                           for key, value in sorted(kwds.iteritems())])
        if cache_key in func._cache:
            func._cache_statistics.hits += 1
        else:
            func._cache_statistics.misses += 1
            func._cache[cache_key] = func(*args, **kwds)
        return func._cache[cache_key]

    return cached_func


def _hashable(obj):
    '''Helper function to make a few data structures hashable (e.g. a
    dictionary gets converted to a frozenset). The function is specifically
    tailored to our use case and not meant to be generally useful.'''
    if hasattr(obj, '_state_tuple'):
        return _hashable(obj._state_tuple)

    try:
        # If the object is already hashable, do nothing
        hash(obj)
        return obj
    except TypeError:
        pass

    if isinstance(obj, set):
        return frozenset(_hashable(el) for el in obj)
    elif isinstance(obj, collections.Sequence):
        return tuple(_hashable(el) for el in obj)
    elif isinstance(obj, collections.Mapping):
        return frozenset((_hashable(key), _hashable(value))
                         for key, value in obj.iteritems())
    else:
        raise AssertionError('Do not know how to handle object of type '
                             '%s' % type(obj))
