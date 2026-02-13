"""
Extension manager for cppyy runtime backend.

This module provides caching, lifecycle management, and utility functions
for the cppyy-based code generation backend. Unlike the Cython extension
manager which deals with filesystem-based compilation, this manager handles
in-memory compilation and function caching.

Key responsibilities:
1. Function cache management (in-memory, content-addressed)
2. Compilation lock management (thread safety)
3. Infrastructure initialization (one-time setup)
4. Diagnostics and statistics

"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from brian2.utils.logger import get_logger

__all__ = [
    "CppyyExtensionManager",
    "CppyyFunctionCache",
    "CppyyInfrastructure",
    "get_extension_manager",
]

logger = get_logger(__name__)


# =============================================================================
# Statistics Tracking
# =============================================================================


@dataclass
class CompilationStats:
    """
    Statistics for cppyy compilation performance.

    Attributes:
        total_compilations: Total number of compilations performed.
        cache_hits: Number of times a cached function was reused.
        cache_misses: Number of times compilation was required.
        total_compile_time: Cumulative time spent compiling (seconds).
        total_code_size: Cumulative size of compiled code (bytes).
        evictions: Number of functions evicted from cache.
    """

    total_compilations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_compile_time: float = 0.0
    total_code_size: int = 0
    evictions: int = 0
    errors: int = 0

    def record_compilation(self, compile_time: float, code_size: int) -> None:
        """Record a successful compilation."""
        self.total_compilations += 1
        self.cache_misses += 1
        self.total_compile_time += compile_time
        self.total_code_size += code_size

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

    def record_eviction(self) -> None:
        """Record a cache eviction."""
        self.evictions += 1

    def record_error(self) -> None:
        """Record a compilation error."""
        self.errors += 1

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def average_compile_time(self) -> float:
        """Calculate average compilation time."""
        if self.total_compilations == 0:
            return 0.0
        return self.total_compile_time / self.total_compilations

    def __str__(self) -> str:
        return (
            f"CompilationStats(\n"
            f"  compilations={self.total_compilations},\n"
            f"  cache_hits={self.cache_hits},\n"
            f"  cache_misses={self.cache_misses},\n"
            f"  hit_rate={self.hit_rate:.1%},\n"
            f"  avg_compile_time={self.average_compile_time * 1000:.1f}ms,\n"
            f"  total_code_size={self.total_code_size / 1024:.1f}KB,\n"
            f"  evictions={self.evictions},\n"
            f"  errors={self.errors}\n"
            f")"
        )


# =============================================================================
# Function Cache
# =============================================================================


@dataclass
class CachedFunction:
    """
    A cached compiled function with metadata.

    Attributes:
        func: The compiled cppyy function proxy.
        code_hash: SHA256 hash of the source code.
        function_name: Name of the C++ function.
        created_at: Timestamp when the function was compiled.
        last_used: Timestamp when the function was last called.
        use_count: Number of times the function has been called.
        code_size: Size of the source code in bytes.
    """

    func: Any
    code_hash: str
    function_name: str
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    code_size: int = 0

    def touch(self) -> None:
        """Update last_used timestamp and increment use count."""
        self.last_used = time.time()
        self.use_count += 1


class CppyyFunctionCache:
    """
    Thread-safe LRU cache for compiled cppyy functions.

    This cache stores compiled C++ functions keyed by a hash of their source
    code. It uses an LRU (Least Recently Used) eviction policy when the cache
    exceeds its maximum size.

    The cache is designed to be shared across all CppyyCodeObject instances
    within a process to maximize code reuse.

    Thread Safety:
        All public methods are thread-safe and protected by a reentrant lock.

    Attributes:
        max_size: Maximum number of functions to cache.
        _cache: OrderedDict mapping code hashes to CachedFunction objects.
        _lock: Threading lock for thread-safe access.
        _stats: Compilation statistics tracker.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """
        Initialize the function cache.

        Args:
            max_size: Maximum number of functions to cache. When exceeded,
                      least recently used functions are evicted.
        """
        self._max_size = max_size
        self._cache: OrderedDict[str, CachedFunction] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CompilationStats()

        # Reverse lookup: function name -> code hash
        self._name_to_hash: dict[str, str] = {}

    @property
    def max_size(self) -> int:
        """Get the maximum cache size."""
        return self._max_size

    @max_size.setter
    def max_size(self, value: int) -> None:
        """Set the maximum cache size, evicting if necessary."""
        with self._lock:
            self._max_size = value
            self._evict_if_needed()

    def get(self, code_hash: str) -> Any | None:
        """
        Get a cached function by its code hash.

        Args:
            code_hash: SHA256 hash of the C++ source code.

        Returns:
            The compiled function proxy, or None if not found.
        """
        with self._lock:
            if code_hash in self._cache:
                cached = self._cache[code_hash]
                cached.touch()
                # Move to end (most recently used)
                self._cache.move_to_end(code_hash)
                self._stats.record_cache_hit()
                return cached.func
            return None

    def put(
        self,
        code_hash: str,
        func: Any,
        function_name: str,
        code_size: int = 0,
    ) -> None:
        """
        Store a compiled function in the cache.

        Args:
            code_hash: SHA256 hash of the C++ source code.
            func: The compiled cppyy function proxy.
            function_name: Name of the C++ function.
            code_size: Size of the source code in bytes.
        """
        with self._lock:
            # Remove existing entry if present
            if code_hash in self._cache:
                old_entry = self._cache.pop(code_hash)
                if old_entry.function_name in self._name_to_hash:
                    del self._name_to_hash[old_entry.function_name]

            # Evict if necessary
            self._evict_if_needed()

            # Add new entry
            cached = CachedFunction(
                func=func,
                code_hash=code_hash,
                function_name=function_name,
                code_size=code_size,
            )
            self._cache[code_hash] = cached
            self._name_to_hash[function_name] = code_hash

    def _evict_if_needed(self) -> None:
        """Evict least recently used entries if cache is over capacity."""
        while len(self._cache) >= self._max_size:
            # Pop the oldest entry (first item in OrderedDict)
            code_hash, cached = self._cache.popitem(last=False)
            if cached.function_name in self._name_to_hash:
                del self._name_to_hash[cached.function_name]
            self._stats.record_eviction()
            logger.debug(f"Evicted cached function: {cached.function_name}")

    def contains(self, code_hash: str) -> bool:
        """Check if a code hash is in the cache."""
        with self._lock:
            return code_hash in self._cache

    def get_by_name(self, function_name: str) -> Any | None:
        """
        Get a cached function by its function name.

        Args:
            function_name: Name of the C++ function.

        Returns:
            The compiled function proxy, or None if not found.
        """
        with self._lock:
            code_hash = self._name_to_hash.get(function_name)
            if code_hash is not None:
                return self.get(code_hash)
            return None

    def clear(self) -> None:
        """Clear all cached functions."""
        with self._lock:
            self._cache.clear()
            self._name_to_hash.clear()
            logger.debug("Function cache cleared")

    def __len__(self) -> int:
        """Return the number of cached functions."""
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> CompilationStats:
        """Get compilation statistics."""
        return self._stats

    def get_info(self) -> dict[str, Any]:
        """
        Get detailed information about the cache state.

        Returns:
            Dictionary with cache information.
        """
        with self._lock:
            entries = []
            for code_hash, cached in self._cache.items():
                entries.append(
                    {
                        "function_name": cached.function_name,
                        "code_hash": code_hash[:16] + "...",
                        "use_count": cached.use_count,
                        "code_size": cached.code_size,
                        "age_seconds": time.time() - cached.created_at,
                    }
                )

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "stats": str(self._stats),
                "entries": entries,
            }


# =============================================================================
# Infrastructure Manager
# =============================================================================


class CppyyInfrastructure:
    """
    Singleton managing cppyy initialization and shared C++ infrastructure.

    This class handles one-time loading of:
    - Standard C++ headers
    - Brian2 common macros and type definitions
    - Random number generator infrastructure
    - DynamicArray and SpikeQueue implementations

    The infrastructure is initialized lazily on first use and shared
    across all CppyyCodeObject instances.

    Thread Safety:
        Initialization is protected by a lock to prevent race conditions.
    """

    _instance: CppyyInfrastructure | None = None
    _initialized: bool = False
    _lock = threading.Lock()

    def __new__(cls) -> CppyyInfrastructure:
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the infrastructure (only runs once)."""
        pass  # Actual init is in ensure_initialized()

    def ensure_initialized(self) -> None:
        """
        Ensure the cppyy infrastructure is initialized.

        This method is idempotent and thread-safe. It only performs
        initialization once, even if called multiple times.
        """
        if CppyyInfrastructure._initialized:
            return

        with CppyyInfrastructure._lock:
            if CppyyInfrastructure._initialized:
                return

            self._load_infrastructure()
            CppyyInfrastructure._initialized = True

    def _load_infrastructure(self) -> None:
        """Load all required C++ infrastructure into cppyy."""
        import cppyy

        logger.debug("Initializing cppyy infrastructure...")
        start_time = time.time()

        # Load standard headers
        cppyy.include("<cmath>")
        cppyy.include("<cstdint>")
        cppyy.include("<vector>")
        cppyy.include("<algorithm>")
        cppyy.include("<random>")
        cppyy.include("<map>")
        cppyy.include("<utility>")

        # Define common infrastructure
        cppyy.cppdef(self._get_common_definitions())
        cppyy.cppdef(self._get_dynamic_array_definitions())
        cppyy.cppdef(self._get_spike_queue_definitions())
        cppyy.cppdef(self._get_random_definitions())

        elapsed = time.time() - start_time
        logger.debug(f"cppyy infrastructure initialized in {elapsed * 1000:.1f}ms")

    def _get_common_definitions(self) -> str:
        """Get common C++ definitions."""
        return """
#ifndef BRIAN2_CPPYY_COMMON
#define BRIAN2_CPPYY_COMMON

#include <cstdint>
#include <cmath>
#include <algorithm>

namespace brian2_cppyy {

// Integer types
using std::int8_t;
using std::int16_t;
using std::int32_t;
using std::int64_t;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::size_t;

// Clip function
template<typename T>
inline T _clip(T value, T min_val, T max_val) {
    return std::min(std::max(value, min_val), max_val);
}

// Integer division (floor division matching Python)
inline int64_t _floordiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a % b;
    if ((r != 0) && ((r < 0) != (b < 0))) {
        q -= 1;
    }
    return q;
}

// Modulo matching Python semantics
inline int64_t _mod(int64_t a, int64_t b) {
    int64_t r = a % b;
    if ((r != 0) && ((r < 0) != (b < 0))) {
        r += b;
    }
    return r;
}

// Sign function
template<typename T>
inline int _sign(T val) {
    return (T(0) < val) - (val < T(0));
}

// Boolean conversion
inline int _bool_to_int(bool b) {
    return b ? 1 : 0;
}

}  // namespace brian2_cppyy

#endif // BRIAN2_CPPYY_COMMON
"""

    def _get_dynamic_array_definitions(self) -> str:
        """Get DynamicArray C++ definitions."""
        return """
#ifndef BRIAN2_CPPYY_DYNAMIC_ARRAY
#define BRIAN2_CPPYY_DYNAMIC_ARRAY

#include <vector>
#include <cstring>
#include <stdexcept>

namespace brian2_cppyy {

template<typename T>
class DynamicArray1D {
private:
    std::vector<T> _data;

public:
    DynamicArray1D() = default;
    explicit DynamicArray1D(size_t size) : _data(size) {}
    DynamicArray1D(size_t size, T value) : _data(size, value) {}

    // Data access
    T* data() noexcept { return _data.data(); }
    const T* data() const noexcept { return _data.data(); }

    // Size operations
    size_t size() const noexcept { return _data.size(); }
    bool empty() const noexcept { return _data.empty(); }
    void resize(size_t new_size) { _data.resize(new_size); }
    void resize(size_t new_size, T value) { _data.resize(new_size, value); }
    void reserve(size_t capacity) { _data.reserve(capacity); }
    void clear() { _data.clear(); }

    // Element access
    T& operator[](size_t idx) { return _data[idx]; }
    const T& operator[](size_t idx) const { return _data[idx]; }
    T& at(size_t idx) { return _data.at(idx); }
    const T& at(size_t idx) const { return _data.at(idx); }

    // Modification
    void push_back(const T& value) { _data.push_back(value); }
    void push_back(T&& value) { _data.push_back(std::move(value)); }
    void pop_back() { _data.pop_back(); }

    // Iterators
    typename std::vector<T>::iterator begin() { return _data.begin(); }
    typename std::vector<T>::iterator end() { return _data.end(); }
    typename std::vector<T>::const_iterator begin() const { return _data.begin(); }
    typename std::vector<T>::const_iterator end() const { return _data.end(); }
};

template<typename T>
class DynamicArray2D {
private:
    std::vector<std::vector<T>> _data;
    size_t _cols;

public:
    DynamicArray2D() : _cols(0) {}
    explicit DynamicArray2D(size_t rows) : _data(rows), _cols(0) {}
    DynamicArray2D(size_t rows, size_t cols)
        : _data(rows, std::vector<T>(cols)), _cols(cols) {}
    DynamicArray2D(size_t rows, size_t cols, T value)
        : _data(rows, std::vector<T>(cols, value)), _cols(cols) {}

    // Row access
    std::vector<T>& operator[](size_t idx) { return _data[idx]; }
    const std::vector<T>& operator[](size_t idx) const { return _data[idx]; }

    // Size operations
    size_t size() const noexcept { return _data.size(); }
    size_t rows() const noexcept { return _data.size(); }
    size_t cols() const noexcept { return _cols; }
    bool empty() const noexcept { return _data.empty(); }

    void resize(size_t new_rows) {
        _data.resize(new_rows);
        for (auto& row : _data) {
            row.resize(_cols);
        }
    }

    void resize(size_t new_rows, size_t new_cols) {
        _cols = new_cols;
        _data.resize(new_rows);
        for (auto& row : _data) {
            row.resize(new_cols);
        }
    }

    void clear() {
        _data.clear();
        _cols = 0;
    }
};

}  // namespace brian2_cppyy

#endif // BRIAN2_CPPYY_DYNAMIC_ARRAY
"""

    def _get_spike_queue_definitions(self) -> str:
        """Get SpikeQueue C++ definitions."""
        return """
#ifndef BRIAN2_CPPYY_SPIKE_QUEUE
#define BRIAN2_CPPYY_SPIKE_QUEUE

#include <vector>
#include <cstdint>
#include <algorithm>

namespace brian2_cppyy {

class SpikeQueue {
private:
    std::vector<std::vector<int32_t>> _queue;
    size_t _current_idx;
    size_t _n_delays;
    int32_t* _delays;
    size_t _n_synapses;
    int32_t _source_start;
    int32_t _source_end;

public:
    SpikeQueue()
        : _current_idx(0)
        , _n_delays(1)
        , _delays(nullptr)
        , _n_synapses(0)
        , _source_start(0)
        , _source_end(0)
    {
        _queue.resize(1);
    }

    SpikeQueue(int32_t source_start, int32_t source_end)
        : _current_idx(0)
        , _n_delays(1)
        , _delays(nullptr)
        , _n_synapses(0)
        , _source_start(source_start)
        , _source_end(source_end)
    {
        _queue.resize(1);
    }

    void prepare(int32_t* delays, size_t n_delays, size_t n_synapses) {
        _delays = delays;
        _n_delays = n_delays > 0 ? n_delays : 1;
        _n_synapses = n_synapses;

        _queue.clear();
        _queue.resize(_n_delays);
        _current_idx = 0;
    }

    void push(int32_t* spike_indices, int n_spikes) {
        if (n_spikes == 0) return;

        // Simple implementation: push all spikes to current slot
        // Full implementation would use delays
        auto& current = _queue[_current_idx % _n_delays];
        for (int i = 0; i < n_spikes; ++i) {
            current.push_back(spike_indices[i]);
        }
    }

    std::vector<int32_t>* peek() {
        return &_queue[_current_idx % _n_delays];
    }

    const std::vector<int32_t>* peek() const {
        return &_queue[_current_idx % _n_delays];
    }

    void advance() {
        _queue[_current_idx % _n_delays].clear();
        _current_idx++;
    }

    size_t size() const {
        return _queue[_current_idx % _n_delays].size();
    }

    void clear() {
        for (auto& slot : _queue) {
            slot.clear();
        }
        _current_idx = 0;
    }
};

}  // namespace brian2_cppyy

#endif // BRIAN2_CPPYY_SPIKE_QUEUE
"""

    def _get_random_definitions(self) -> str:
        """Get random number generator C++ definitions."""
        return """
#ifndef BRIAN2_CPPYY_RANDOM
#define BRIAN2_CPPYY_RANDOM

#include <random>
#include <cstdint>

namespace brian2_cppyy {

// Thread-local random engine for each compilation unit
thread_local std::mt19937_64 _rng;
thread_local bool _rng_seeded = false;

inline void seed_rng(uint64_t seed) {
    _rng.seed(seed);
    _rng_seeded = true;
}

inline void ensure_rng_seeded() {
    if (!_rng_seeded) {
        std::random_device rd;
        _rng.seed(rd());
        _rng_seeded = true;
    }
}

inline double _rand() {
    ensure_rng_seeded();
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(_rng);
}

inline double _randn() {
    ensure_rng_seeded();
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(_rng);
}

inline int32_t _poisson(double lambda) {
    ensure_rng_seeded();
    if (lambda <= 0) return 0;
    std::poisson_distribution<int32_t> dist(lambda);
    return dist(_rng);
}

inline double _rand_uniform(double low, double high) {
    ensure_rng_seeded();
    std::uniform_real_distribution<double> dist(low, high);
    return dist(_rng);
}

inline int32_t _rand_int(int32_t low, int32_t high) {
    ensure_rng_seeded();
    std::uniform_int_distribution<int32_t> dist(low, high - 1);
    return dist(_rng);
}

inline double _rand_exponential(double beta) {
    ensure_rng_seeded();
    std::exponential_distribution<double> dist(1.0 / beta);
    return dist(_rng);
}

inline double _rand_gamma(double alpha, double beta) {
    ensure_rng_seeded();
    std::gamma_distribution<double> dist(alpha, beta);
    return dist(_rng);
}

}  // namespace brian2_cppyy

#endif // BRIAN2_CPPYY_RANDOM
"""

    @property
    def is_initialized(self) -> bool:
        """Check if infrastructure is initialized."""
        return CppyyInfrastructure._initialized

    @classmethod
    def reset(cls) -> None:
        """
        Reset the infrastructure state (mainly for testing).

        Warning: This does not unload the C++ definitions from cppyy,
        as that's not possible. It only resets the initialization flag.
        """
        with cls._lock:
            cls._initialized = False
            cls._instance = None


# =============================================================================
# Extension Manager
# =============================================================================


class CppyyExtensionManager:
    """
    Central manager for the cppyy runtime backend.

    This class coordinates:
    - Function caching
    - Infrastructure initialization
    - Compilation with thread safety
    - Statistics and diagnostics

    It provides a high-level interface for CppyyCodeObject to use
    for compiling and caching C++ code.
    """

    _instance: CppyyExtensionManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> CppyyExtensionManager:
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the extension manager."""
        if self._initialized:
            return

        from brian2.core.preferences import prefs

        # Get cache size from preferences (with fallback)
        try:
            cache_size = prefs["codegen.runtime.cppyy.cache_size"]
        except KeyError:
            cache_size = 1000

        self._cache = CppyyFunctionCache(max_size=cache_size)
        self._infrastructure = CppyyInfrastructure()
        self._compile_lock = threading.Lock()
        self._initialized = True

    @property
    def cache(self) -> CppyyFunctionCache:
        """Get the function cache."""
        return self._cache

    @property
    def infrastructure(self) -> CppyyInfrastructure:
        """Get the infrastructure manager."""
        return self._infrastructure

    @property
    def stats(self) -> CompilationStats:
        """Get compilation statistics."""
        return self._cache.stats

    def ensure_initialized(self) -> None:
        """Ensure infrastructure is initialized."""
        self._infrastructure.ensure_initialized()

    def compile(
        self,
        code: str,
        function_name: str,
        force: bool = False,
    ) -> Any:
        """
        Compile C++ code and return the function proxy.

        This method handles:
        1. Computing code hash
        2. Checking cache
        3. Thread-safe compilation
        4. Caching the result

        Args:
            code: C++ source code to compile.
            function_name: Name of the function in the code.
            force: If True, recompile even if cached.

        Returns:
            The compiled cppyy function proxy.

        Raises:
            RuntimeError: If compilation fails.
        """
        import cppyy

        # Ensure infrastructure is ready
        self.ensure_initialized()

        # Compute hash
        code_hash = hashlib.sha256(code.encode()).hexdigest()

        # Check cache (unless forced)
        if not force:
            cached = self._cache.get(code_hash)
            if cached is not None:
                logger.debug(f"Cache hit for {function_name}")
                return cached

        # Compile with lock
        with self._compile_lock:
            # Double-check cache (another thread may have compiled)
            if not force:
                cached = self._cache.get(code_hash)
                if cached is not None:
                    return cached

            logger.debug(f"Compiling {function_name}...")
            start_time = time.time()

            try:
                cppyy.cppdef(code)
                func = getattr(cppyy.gbl, function_name)
            except Exception as e:
                self._cache.stats.record_error()
                raise RuntimeError(
                    f"Failed to compile {function_name}: {e}\n"
                    f"Code:\n{self._format_code(code)}"
                ) from e

            compile_time = time.time() - start_time
            code_size = len(code.encode())

            # Update stats
            self._cache.stats.record_compilation(compile_time, code_size)

            # Cache the result
            self._cache.put(code_hash, func, function_name, code_size)

            logger.debug(
                f"Compiled {function_name} in {compile_time * 1000:.1f}ms "
                f"({code_size} bytes)"
            )

            return func

    def _format_code(self, code: str, max_lines: int = 50) -> str:
        """Format code with line numbers for error messages."""
        lines = code.split("\n")
        if len(lines) > max_lines:
            # Show first and last portions
            half = max_lines // 2
            formatted_lines = []
            for i, line in enumerate(lines[:half], 1):
                formatted_lines.append(f"{i:4d} | {line}")
            formatted_lines.append(
                f"     | ... ({len(lines) - max_lines} lines omitted) ..."
            )
            for i, line in enumerate(lines[-half:], len(lines) - half + 1):
                formatted_lines.append(f"{i:4d} | {line}")
            return "\n".join(formatted_lines)
        else:
            return "\n".join(f"{i:4d} | {line}" for i, line in enumerate(lines, 1))

    def clear_cache(self) -> None:
        """Clear the function cache."""
        self._cache.clear()

    def get_diagnostics(self) -> dict[str, Any]:
        """
        Get diagnostic information about the extension manager.

        Returns:
            Dictionary with diagnostic information.
        """
        return {
            "infrastructure_initialized": self._infrastructure.is_initialized,
            "cache_info": self._cache.get_info(),
            "stats": str(self._cache.stats),
        }


# =============================================================================
# Module-level Access
# =============================================================================

# Global extension manager instance
_extension_manager: CppyyExtensionManager | None = None


def get_extension_manager() -> CppyyExtensionManager:
    """
    Get the global extension manager instance.

    Returns:
        The singleton CppyyExtensionManager instance.
    """
    global _extension_manager
    if _extension_manager is None:
        _extension_manager = CppyyExtensionManager()
    return _extension_manager
