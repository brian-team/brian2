# Brian2 cppyy JIT Codegen Backend

## Overview

The cppyy backend replaces the Cython AOT (ahead-of-time) compilation pipeline with cppyy/Cling JIT (just-in-time) compilation. Instead of generating `.pyx` files, invoking an external C++ compiler, and loading `.so` shared libraries, the cppyy backend compiles C++ strings to machine code at runtime via LLVM — no external compiler required.

**Why this matters**: Cython compilation adds 30-60 seconds of startup overhead per simulation. The cppyy backend eliminates this entirely while running the same C++ code at the same speed.

```mermaid
graph LR
    subgraph "Cython Pipeline (AOT)"
        A1[Equations] --> B1[.pyx source]
        B1 --> C1[C compiler]
        C1 --> D1[.so library]
        D1 --> E1[Execute]
    end

    subgraph "cppyy Pipeline (JIT)"
        A2[Equations] --> B2[C++ string]
        B2 --> C2[cppyy/Cling/LLVM]
        C2 --> E2[Execute]
    end

    style C1 fill:#f96,stroke:#333
    style C2 fill:#6f9,stroke:#333
```

---

## End-to-End Flow: From Equations to Execution

This diagram shows the complete lifecycle from user-written equations to JIT-compiled machine code:

```mermaid
flowchart TD
    USER["User Code<br/><code>NeuronGroup(N, 'dv/dt = ...')</code>"]
    --> NETWORK["Network.before_run()"]
    --> ABSTRACT["AbstractCodeObject.create_runner_codeobj()"]
    --> GENERATOR["CppyyCodeGenerator"]
    --> TEMPLATE["Jinja2 Template<br/>(e.g., stateupdate.cpp)"]
    --> RENDERED["Rendered C++ string<br/>extern 'C' void _brian_cppyy_run_...()"]
    --> CPPDEF["cppyy.cppdef(code)<br/>LLVM JIT compilation"]
    --> FUNCPTR["Function pointer<br/>cppyy.gbl._brian_cppyy_run_..."]
    --> CALL["run_block() calls func(*args)<br/>numpy arrays → C++ pointers<br/>(zero-copy)"]

    subgraph "Per timestep"
        CALL --> UPDATE["update_namespace()<br/>refresh dynamic array pointers"]
        UPDATE --> CALL
    end

    style CPPDEF fill:#6f9,stroke:#333
    style CALL fill:#69f,stroke:#333
```

---

## Target Selection and Backend Priorities

When Brian2 starts, it registers available codegen targets:

```mermaid
flowchart TD
    AUTO["prefs.codegen.target = 'auto'"]
    --> CHECK_CYTHON{"Cython +<br/>C++ compiler?"}
    CHECK_CYTHON -->|Yes| USE_CYTHON["Use cython"]
    CHECK_CYTHON -->|No| CHECK_CPPYY{"cppyy<br/>installed?"}
    CHECK_CPPYY -->|Yes| USE_CPPYY["Use cppyy"]
    CHECK_CPPYY -->|No| USE_NUMPY["Use numpy<br/>(pure Python fallback)"]

    FORCE["prefs.codegen.target = 'cppyy'"]
    --> USE_CPPYY_DIRECT["Use cppyy directly"]

    style USE_CPPYY fill:#6f9,stroke:#333
    style USE_CPPYY_DIRECT fill:#6f9,stroke:#333
```

Each target registers via `codegen_targets.add(MyCodeObject)` in its `__init__.py`. The `auto_target()` function in `device.py` picks the best available.

---

## The Three Naming Worlds

The most critical invariant in the codebase. Three different naming conventions must stay synchronized across the Python/C++ boundary:

```mermaid
flowchart LR
    subgraph "1. Device Layer (Python)"
        DEV["RuntimeDevice stores:<br/><code>_array_neurongroup_v</code>"]
    end

    subgraph "2. Generator Layer (C++ params)"
        GEN["Function signature:<br/><code>double* _ptr_array_neurongroup_v</code>"]
    end

    subgraph "3. C++ Body"
        BODY["Loop body:<br/><code>_ptr_array_neurongroup_v[_idx]</code>"]
    end

    DEV -->|"variables_to_namespace()<br/>copies under generator names"| GEN
    GEN -->|"Same name used<br/>in template code"| BODY
```

| Layer | Static Array | Dynamic Array (data) | Dynamic Array (container) |
|-------|-------------|----------------------|---------------------------|
| Device | `_array_G_v` | `_array_G_v` | `_dynamic_array_G_v` |
| Generator | `_ptr_array_G_v` | `_ptr_array_G_v` | `_dynamic_array_G_v` |
| C++ body | `_ptr_array_G_v[_idx]` | `_ptr_array_G_v[_idx]` | via `_dynamic_array_G_v_capsule` |

The generator's `get_array_name()` produces names for layers 2 and 3. The code object's `variables_to_namespace()` bridges layer 1 → 2.

---

## Parameter Synchronization: The Critical Invariant

The C++ function signature and the Python call-site arguments **must** have identical parameter order. This is maintained by two functions that mirror each other:

```mermaid
flowchart TD
    subgraph "Code Generation Time"
        DK["CppyyCodeGenerator.determine_keywords()<br/>Iterates sorted(variables.items())<br/>Builds function_params for Jinja2"]
    end

    subgraph "Runtime"
        BPM["CppyyCodeObject._build_param_mapping()<br/>Iterates sorted(self.variables.items())<br/>Builds (cpp_name, ns_key, c_type) tuples"]
    end

    subgraph "Execution"
        RB["run_block()<br/>Reads namespace values<br/>in param_mapping order<br/>Calls func(*args)"]
    end

    DK -->|"Same iteration order<br/>Same filtering logic<br/>Same param additions"| BPM
    BPM --> RB

    style DK fill:#ff9,stroke:#333
    style BPM fill:#ff9,stroke:#333
```

**Both functions MUST**:
1. Iterate `sorted(self.variables.items())`
2. Skip `AuxiliaryVariable`, `Subexpression`, `Function`
3. Handle `Constant` → scalar param
4. Handle `ArrayVariable` → pointer + `_num*` size + capsule (for dynamic)
5. Handle `*_capsule` object variables (e.g., `_queue_capsule`)

If these diverge, the C++ function receives wrong arguments → segfault or silent corruption.

---

## Template Architecture

All templates extend `common_group.cpp`, which defines the `extern "C"` function skeleton with three blocks:

```mermaid
flowchart TD
    subgraph "common_group.cpp"
        BEFORE["{% macro before_run() %}<br/>extern 'C' void _brian_cppyy_before_run_...(params)<br/>{% block before_code %} EMPTY {% endblock %}"]
        RUN["{% macro run() %}<br/>{{ hashdefine_lines }}<br/>{{ support_code_lines }}<br/>{% block template_support_code %}{% endblock %}<br/>extern 'C' void _brian_cppyy_run_...(params)<br/>{% block maincode %}{% endblock %}"]
        AFTER["{% macro after_run() %}<br/>extern 'C' void _brian_cppyy_after_run_...(params)<br/>{% block after_code %} EMPTY {% endblock %}"]
    end

    subgraph "stateupdate.cpp"
        SU["{% block maincode %}<br/>for (_idx = 0; _idx < N; _idx++)<br/>&nbsp;&nbsp;{{ vector_code }}<br/>{% endblock %}"]
    end

    subgraph "synapses.cpp"
        SYN["{% block template_support_code %}<br/>#include &lt;vector&gt;<br/>{% endblock %}<br/><br/>{% block maincode %}<br/>CSpikeQueue* _queue = ...<br/>for each spike: {{ vector_code }}<br/>{% endblock %}"]
    end

    RUN --> SU
    RUN --> SYN

    style RUN fill:#e9e,stroke:#333
```

### Template Inventory

| Template | Purpose | Uses Capsules? | Special Features |
|----------|---------|----------------|------------------|
| `stateupdate.cpp` | ODE integration | No | Simple N-loop |
| `threshold.cpp` | Spike detection | No | Writes to spikespace |
| `reset.cpp` | Post-spike reset | No | Conditional on spikespace |
| `spikemonitor.cpp` | Record spike times | Yes (1D) | Resizes `t`, `i`, recorded vars |
| `statemonitor.cpp` | Record variable traces | Yes (2D) | Resizes 2D arrays per timestep |
| `ratemonitor.cpp` | Population rate | Yes (1D) | Resizes `t`, `rate` |
| `synapses.cpp` | Synapse propagation | Yes (SpikeQueue) | Extracts queue, peeks, advances |
| `synapses_push_spikes.cpp` | Push spikes to queue | Yes (SpikeQueue) | Reads eventspace |
| `synapses_create_array.cpp` | Direct i,j creation | Yes (1D) | Resizes pre/post arrays |
| `synapses_create_generator.cpp` | Generator-based creation | Yes (1D) | Buffered (1024) resize pattern |
| `summed_variable.cpp` | Summed synaptic vars | No | Accumulates across synapses |
| `group_variable_get.cpp` | Get variable values | No | |
| `group_variable_set.cpp` | Set variable values | No | |
| `group_variable_get_conditional.cpp` | Conditional get | No | |
| `group_variable_set_conditional.cpp` | Conditional set | No | |

**Missing** (still need porting from Cython):
- `spatialstateupdate` — multi-compartment neurons (complex linear algebra)
- `spikegenerator` — SpikeGeneratorGroup (straightforward)
- `group_get_indices` — group index operations (low priority)

---

## Zero-Copy Data Bridge

The cppyy backend achieves zero-copy data transfer between Python and C++:

```mermaid
flowchart LR
    subgraph "Python Side"
        NP["numpy array<br/>np.float64, contiguous"]
        CAP["PyCapsule<br/>wraps DynamicArray1D&lt;T&gt;*"]
    end

    subgraph "cppyy Buffer Protocol"
        CONV["Automatic conversion<br/>np.ndarray → T*<br/>(no copy)"]
    end

    subgraph "C++ Side"
        PTR["double* _ptr_array_G_v<br/>Direct memory access"]
        DYN["DynamicArray1D&lt;T&gt;*<br/>resize(), get_data_ptr()"]
    end

    NP --> CONV --> PTR
    CAP -->|"PyCapsule_GetPointer()"| DYN

    style CONV fill:#6f9,stroke:#333
```

### Two Kinds of Array Access

**Static arrays** (fixed size, e.g., neuron state variables):
```
Python: np.ndarray → cppyy → double* (zero copy)
C++:    _ptr_array_G_v[_idx] = ...
```

**Dynamic arrays** (resizable, e.g., monitor recordings):
```
Python: PyCapsule → cppyy → DynamicArray1D<T>*
C++:    _dyn->resize(_newlen);
        _dyn->get_data_ptr()[i] = value;
```

PyCapsule names are standardized (`"DynamicArray1D"`, `"DynamicArray2D"`, `"CSpikeQueue"`) and extracted with templated helpers defined in the global support code.

---

## DynamicArray Backend

`brian2/memory/cppyy_dynamicarray.py` is a drop-in replacement for the Cython DynamicArray wrappers:

```mermaid
flowchart TD
    subgraph "Import Chain"
        DA["brian2/memory/dynamicarray.py"]
        -->|"try:"| CY["cythondynamicarray.pyx<br/>(Cython wrapper)"]
        DA -->|"except: try:"| CP["cppyy_dynamicarray.py<br/>(cppyy wrapper)"]
        DA -->|"except:"| FAIL["ImportError"]
    end

    subgraph "cppyy_dynamicarray.py Internals"
        INIT["_ensure_initialized()<br/>cppyy.include('dynamic_array.h')"]
        --> INST["cppyy.gbl.DynamicArray1D['double']()<br/>Instantiates C++ template"]
        --> CAPSULE["PyCapsule_New(ptr, 'DynamicArray1D', NULL)<br/>via ctypes.pythonapi"]
        --> VIEW["numpy view via<br/>ctypes.from_address() + np.ctypeslib.as_array()"]
    end

    style CP fill:#6f9,stroke:#333
```

Key design choices:
- Uses `ctypes.pythonapi.PyCapsule_New` instead of cppyy for capsule creation (avoids template lookup issues)
- Uses C++ helpers returning `uintptr_t` for reliable address extraction (`cppyy.addressof()` fails on some pointer types)
- numpy views are zero-copy via `ctypes.from_address()`

---

## SpikeQueue Backend

`brian2/synapses/cppyy_spikequeue.py` wraps `CSpikeQueue` from `spikequeue.h`:

```mermaid
flowchart TD
    subgraph "Import Chain"
        SQ["brian2/synapses/spikequeue.py"]
        -->|"try:"| CY["cspikequeue.pyx<br/>(Cython wrapper)"]
        SQ -->|"except: try:"| CP["cppyy_spikequeue.py<br/>(cppyy wrapper)"]
    end

    subgraph "C++ Helper Functions"
        PEEK["_brian_sq_peek_data()<br/>Returns raw pointer + size"]
        PUSH["_brian_sq_push_array()<br/>Pushes spike indices by pointer"]
        PREP["_brian_sq_prepare&lt;T&gt;()<br/>Template for float/double delays"]
    end

    subgraph "Spike Propagation Flow"
        PRE_SPIKES["Pre-neuron spikes<br/>(eventspace)"]
        -->|"synapses_push_spikes.cpp"| QUEUE["CSpikeQueue<br/>(ring buffer of delay slots)"]
        -->|"synapses.cpp"| POST["peek() → synapse indices<br/>Run pre/post code<br/>advance()"]
    end

    CP --> PEEK
    CP --> PUSH
    CP --> PREP
```

---

## Synapse Creation and Lifecycle

Synapse creation is the most complex part of the backend:

```mermaid
sequenceDiagram
    participant User as User Code
    participant Synapses as Synapses.connect()
    participant CodeObj as CppyyCodeObject
    participant Cling as Cling JIT
    participant DynArray as DynamicArray1D<int32_t>

    User->>Synapses: S.connect(j='i')
    Synapses->>CodeObj: create runner codeobj<br/>(synapses_create_generator.cpp)
    CodeObj->>Cling: cppyy.cppdef(rendered_code)
    Cling-->>CodeObj: function pointer

    Note over CodeObj: Pass capsules for<br/>_synaptic_pre, _synaptic_post

    CodeObj->>Cling: func(capsules, arrays, ...)

    loop For each pre/post pair
        Cling->>DynArray: Buffer synapse indices<br/>(1024-element buffer)
        Note over Cling,DynArray: Flush when buffer full:<br/>resize() + memcpy()
    end

    Cling->>DynArray: Final flush of remaining

    Cling-->>CodeObj: Return

    Note over Synapses: Python-side bookkeeping:<br/>_resize(len(S))<br/>_update_synapse_numbers()
```

### Buffered Synapse Creation (Performance Optimization)

The generator template uses a 1024-element buffer to batch resize calls:

```
Without buffering: O(n) resize calls     (1 per synapse)
With buffering:    O(n/1024) resize calls (1 per 1024 synapses)
```

This matches the Cython backend's batching pattern.

---

## Monitor Data Flow

All three monitors follow the same capsule-based pattern:

```mermaid
flowchart TD
    subgraph "SpikeMonitor"
        SM_SPIKE["Detect spikes<br/>(from eventspace)"]
        --> SM_RESIZE["Resize via capsules:<br/>t, i, recorded vars"]
        --> SM_CACHE["Cache data pointers<br/>(extracted once)"]
        --> SM_WRITE["Write spike data<br/>using cached pointers"]
    end

    subgraph "StateMonitor"
        ST_TIME["Append current time<br/>(1D resize via capsule)"]
        --> ST_RESIZE["Resize 2D arrays<br/>(via capsule)"]
        --> ST_CACHE["Cache 2D array refs<br/>(extracted once)"]
        --> ST_WRITE["For each neuron:<br/>write via cached ref"]
    end

    subgraph "RateMonitor"
        RM_COUNT["Count spikes in<br/>source range"]
        --> RM_RESIZE["Resize t, rate<br/>(via capsules)"]
        --> RM_WRITE["Write time + rate"]
    end
```

The capsule pattern means the C++ code directly manipulates the same `DynamicArray` objects that Cython created — no Python overhead for resize/write.

---

## Guard-Protected Support Code

Cling (cppyy's compiler) can't redefine symbols. When Brian2 recreates code objects (e.g., calling `run()` multiple times), identical support code would cause redefinition errors:

```mermaid
flowchart TD
    CODE["Generated C++ code"]
    --> SPLIT["Split at 'extern \"C\"'"]

    SPLIT --> SUPPORT["Support code<br/>(inline functions, #defines)"]
    SPLIT --> FUNC["Function definition<br/>(unique name per codeobject)"]

    SUPPORT --> HASH["MD5 hash of<br/>non-comment lines"]
    HASH --> GUARD["#ifndef _BRIAN_CPPYY_SC_{hash}<br/>#define ...<br/>...support code...<br/>#endif"]

    GUARD --> MERGE["Merge back with<br/>function definition"]
    FUNC --> MERGE
    MERGE --> CPPDEF["cppyy.cppdef()"]
```

The `_guard_support_code()` function in `cppyy_rt.py` handles this automatically. The function definition (with its unique `_brian_cppyy_{block}_{name}` identifier) is left unguarded.

---

## Global Support Code

One-time initialization compiled via `_ensure_support_code()`:

```mermaid
flowchart TD
    ENSURE["_ensure_support_code()"]
    --> HEADERS["#include &lt;cmath&gt;, &lt;random&gt;, etc."]
    --> BRIANLIB["#include 'dynamic_array.h'<br/>#include 'spikequeue.h'"]
    --> UNIVERSAL["_brian_mod, _brian_pow,<br/>_brian_floordiv, etc."]
    --> INT["int_() template"]
    --> RNG["std::mt19937 _brian_cppyy_rng<br/>_brian_cppyy_seed()<br/>_brian_cppyy_seed_random()"]
    --> EXTRACT["_extract_dynamic_array_1d&lt;T&gt;()<br/>_extract_dynamic_array_2d&lt;T&gt;()<br/>_extract_spike_queue()"]

    ENSURE -->|"All inside<br/>#ifndef _BRIAN2_CPPYY_SUPPORT_CODE"| GUARDED["Compiled exactly once<br/>per cppyy session"]

    style GUARDED fill:#6f9,stroke:#333
```

---

## Compilation and Execution Lifecycle

```mermaid
sequenceDiagram
    participant Network as Network
    participant CodeObj as CppyyCodeObject
    participant Generator as CppyyCodeGenerator
    participant Template as Jinja2 Template
    participant Cling as Cling JIT
    participant Func as C++ Function

    Note over Network: before_run phase

    Network->>CodeObj: compile()
    CodeObj->>Generator: translate code blocks
    Generator->>Template: render with keywords
    Template-->>CodeObj: C++ source string

    CodeObj->>CodeObj: _guard_support_code()
    CodeObj->>Cling: cppyy.cppdef(guarded_code)
    Cling-->>CodeObj: OK

    CodeObj->>Cling: getattr(cppyy.gbl, func_name)
    Cling-->>CodeObj: function pointer

    CodeObj->>CodeObj: _build_param_mapping()
    CodeObj->>CodeObj: _set_user_func_globals()

    Note over Network: run phase (per timestep)

    loop Every timestep
        Network->>CodeObj: run_block('run')
        CodeObj->>CodeObj: update_namespace()<br/>(refresh dynamic array ptrs)
        CodeObj->>CodeObj: Build args from<br/>param_mapping + namespace
        CodeObj->>Func: func(*args)
        Func-->>CodeObj: return
    end
```

---

## Architecture: File Map

```
brian2/
├── codegen/
│   ├── _prefs.py                       # codegen.target preference ("auto"/"cppyy"/...)
│   ├── targets.py                      # codegen_targets registry (set of CodeObject classes)
│   ├── generators/
│   │   ├── cpp_generator.py            # Base C++ translation (expressions, statements)
│   │   └── cppyy_generator.py          # cppyy overrides: array naming, parameter assembly
│   └── runtime/
│       ├── __init__.py                 # Imports and registers all targets
│       └── cppyy_rt/
│           ├── __init__.py             # Registers CppyyCodeObject
│           ├── cppyy_rt.py             # CppyyCodeObject: compile, namespace, run
│           ├── introspector.py         # Optional runtime inspection tool
│           └── templates/
│               ├── common_group.cpp    # Base template: extern "C" function skeleton
│               ├── stateupdate.cpp     # ODE integration loop
│               ├── threshold.cpp       # Spike detection
│               ├── reset.cpp           # Post-spike reset
│               ├── ratemonitor.cpp     # Population rate recording
│               ├── spikemonitor.cpp    # Individual spike recording
│               ├── statemonitor.cpp    # Variable trace recording
│               ├── synapses.cpp        # Synapse propagation
│               ├── synapses_push_spikes.cpp
│               ├── synapses_create_array.cpp
│               ├── synapses_create_generator.cpp
│               ├── summed_variable.cpp
│               ├── group_variable_get.cpp
│               ├── group_variable_get_conditional.cpp
│               ├── group_variable_set.cpp
│               └── group_variable_set_conditional.cpp
├── devices/
│   └── device.py                       # auto_target(), seed() with cppyy RNG support
├── memory/
│   ├── dynamicarray.py                 # Fallback chain: Cython → cppyy
│   └── cppyy_dynamicarray.py           # cppyy DynamicArray wrapper
└── synapses/
    ├── spikequeue.py                   # Fallback chain: Cython → cppyy
    ├── cppyy_spikequeue.py             # cppyy SpikeQueue wrapper
    └── synapses.py                     # Python-side synapse bookkeeping for cppyy
```

---

## What Was Changed and Why

### Phase 1: Quick Wins

**ratemonitor.cpp** — Was completely broken. Used nonexistent `.push_back()` on DynamicArray. Rewritten to use capsule-based resize pattern matching spikemonitor/statemonitor.

**RNG seeding** — Added `_brian_cppyy_seed()` and `_brian_cppyy_seed_random()` C++ functions exposed via `extern "C"`. Called from `device.py:seed()` so `seed(42)` works for reproducible simulations.

**Parameter assertions** — Added diagnostic logging in `run_block()` to catch parameter count mismatches between Python and C++ sides.

### Phase 2: DynamicArray Backend

**`brian2/memory/cppyy_dynamicarray.py`** — Drop-in cppyy replacement for Cython's DynamicArray wrappers. Uses:
- `cppyy.gbl.DynamicArray1D["double"]()` to instantiate C++ templates
- `ctypes.pythonapi.PyCapsule_New` for capsule creation (ctypes, not cppyy — avoids template lookup issues)
- C++ helpers `_brian_dynarray_data_addr_1d/2d` returning `uintptr_t` for reliable address extraction
- Zero-copy numpy views via `ctypes.from_address()` + `np.ctypeslib.as_array()`

**`brian2/memory/dynamicarray.py`** — Changed from hard Cython import to fallback chain: Cython → cppyy.

### Phase 3: SpikeQueue

**`brian2/synapses/cppyy_spikequeue.py`** — cppyy wrapper for CSpikeQueue from `spikequeue.h`. Uses C++ helpers for data passing:
- `_brian_sq_peek_data()` — returns raw pointer + size for zero-copy peek
- `_brian_sq_push_array()` — pushes spike indices by pointer
- `_brian_sq_prepare<scalar>()` — template function for float/double delay preparation

**`brian2/synapses/spikequeue.py`** — Same Cython → cppyy fallback pattern.

### Phase 4: Synapse Templates

Four new templates in `brian2/codegen/runtime/cppyy_rt/templates/`:

- **synapses.cpp** — Main synapse propagation: extracts CSpikeQueue from `_queue_capsule`, peeks synapse indices, runs pre/post code per synapse, advances queue
- **synapses_push_spikes.cpp** — Reads spike count from eventspace, pushes to queue
- **synapses_create_array.cpp** — Direct i,j synapse creation via capsule-based resize
- **synapses_create_generator.cpp** — Complex template handling range, fixed-sample, and probabilistic iterators

**Key fix**: `_queue_capsule` (an `ObjectVariable`) wasn't being included in function parameters because the generator/code object only handled `ArrayVariable`, `Constant`, and `Function`. Added detection of `*_capsule`-named non-standard variables in both `determine_keywords()` and `_build_param_mapping()`.

**Key fix**: After cppyy creates synapses (resizing DynamicArrays in C++), Python-side `N` and synapse bookkeeping wasn't updated. Added explicit `_owner._resize()` / `_owner._update_synapse_numbers()` calls from Python after the code object runs.

### Phase 5: Performance Audit

**Buffered synapse creation** — `synapses_create_generator.cpp` now uses 1024-element buffers with `_flush_buffer()` helper (O(n/1024) resize calls, matching Cython).

**Cached capsule extraction** — `spikemonitor.cpp` and `statemonitor.cpp` extract capsules once before loops instead of per-iteration.

**Consolidated helpers** — `_extract_spike_queue()` moved from template support code to global support code (defined once, available everywhere).

**Standardized naming** — `ratemonitor.cpp` now uses `get_array_name()` instead of hardcoded `_dynamic_array_` prefix.

### Phase 6: Multi-Backend Support

Already clean — no additional changes needed:
- `auto_target()` handles priority selection
- Fallback chains in `dynamicarray.py` and `spikequeue.py`
- All three targets (numpy, cython, cppyy) coexist cleanly

---

## How to Experiment

### Basic Setup

```python
from brian2 import *
prefs.codegen.target = "cppyy"
```

### Running the Test Suite

```bash
cd /path/to/brian2
source venv/bin/activate

# Comprehensive audit test (16 tests, subprocess-isolated)
python test-cppyy-audit.py

# Basic HH neuron test
python test-cppyy.py

# Synapse tests (connectivity, STDP, summed variables)
python test-cppyy-synapses.py

# DynamicArray backend tests
python test-cppyy-dynarray.py
```

### Introspection

Enable to inspect generated C++ code at runtime:

```python
prefs.codegen.runtime.cppyy.enable_introspection = True

# After running a simulation:
from brian2.codegen.runtime.cppyy_rt.introspector import get_introspector
intro = get_introspector()
intro.summary()  # Shows all compiled code objects
intro.source("neurongroup_stateupdater")  # View generated C++ source
```

### Preferences

```python
# Extra compiler flags for Cling
prefs.codegen.runtime.cppyy.extra_compile_args = ['-O2', '-ffast-math']

# Enable introspection
prefs.codegen.runtime.cppyy.enable_introspection = True
```

---

## Limitations and Open Problems

### Missing Templates

Three Cython templates have no cppyy equivalent yet:
- **`spatialstateupdate`** — Spatial (multi-compartment) neurons. Uses specialized linear algebra that needs careful porting.
- **`spikegenerator`** — SpikeGeneratorGroup. Relatively straightforward to port.
- **`group_get_indices`** — Used by some group operations. Low priority.

### Known Issues

1. **First-run JIT overhead**: The first `cppyy.cppdef()` call takes ~2-3 seconds as Cling initializes. Subsequent calls are fast. This is still much faster than Cython's full compilation cycle.

2. **Cling memory**: Cling keeps all compiled code in memory. Very long sessions with many `run()` calls accumulate compiled code. Not a problem for typical use.

3. **Template instantiation**: cppyy's template lookup syntax (`cppyy.gbl.func["type"]`) can be finicky. We use C++ helper functions with explicit types to work around this.

4. **`addressof` limitations**: `cppyy.addressof()` fails on certain pointer types (e.g., `char*`). We work around this with C++ helper functions that return `uintptr_t`.

5. **No standalone device support**: The cppyy backend only works with `RuntimeDevice`, not `CPPStandaloneDevice`. The standalone device generates complete C++ projects — a fundamentally different approach.

6. **Cling state conflicts**: Multiple `start_scope()` calls within a single process can cause Cling redefinition errors. The guard code handles most cases, but edge cases exist. Test suites should use subprocess isolation.

7. **Time-based refractoriness**: `refractory=5*ms` may not enforce exact timing with some integration methods. String-based conditions (`refractory='v > -40*mV'`) work correctly.

---

## Continuation Prompt

Use this to resume work in a new conversation:

> I'm working on the brian2 `experimental-cppyy` branch — a cppyy/Cling JIT codegen backend that replaces Cython AOT compilation. The backend is functional with: state updates, thresholds, resets, monitors (spike/state/rate), synapses (creation, propagation, STDP), DynamicArray/SpikeQueue fallbacks, RNG seeding, and an introspector.
>
> Key files: `brian2/codegen/runtime/cppyy_rt/cppyy_rt.py` (code object), `brian2/codegen/generators/cppyy_generator.py` (code generator), templates in `brian2/codegen/runtime/cppyy_rt/templates/`, `brian2/memory/cppyy_dynamicarray.py`, `brian2/synapses/cppyy_spikequeue.py`.
>
> Read `docs/cppyy-backend.md` for the full architecture doc. The three naming worlds (device → generator → C++ body) and parameter sync between `determine_keywords()` and `_build_param_mapping()` are the most critical invariants.
>
> Next steps: port missing templates (spatialstateupdate, spikegenerator, group_get_indices), and run the full Brian2 test suite with `prefs.codegen.target = "cppyy"`.
