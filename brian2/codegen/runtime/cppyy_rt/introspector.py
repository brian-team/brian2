"""
Runtime introspection for the cppyy backend.

Provides a live window into the C++ JIT interpreter during simulation.
Enable with: prefs.codegen.runtime.cppyy.enable_introspection = True

Usage in a Jupyter notebook or script:

    from brian2.codegen.runtime.cppyy_rt.introspector import get_introspector
    intro = get_introspector()

    intro.list_objects()                          # see all compiled code objects
    intro.source("neurongroup_stateupdater_*")    # view generated C++
    intro.params("neurongroup_stateupdater_*")    # view parameter mapping
    intro.namespace("neurongroup_stateupdater_*") # view runtime values

    body = intro.get_body("neurongroup_stateupdater_*", "run")
    new_body = body.replace("exp(_lio_2)", "1.0 + _lio_2")
    intro.replace_body("neurongroup_stateupdater_*", "run", new_body)

    intro.restore("neurongroup_stateupdater_*", "run")  # undo
"""

from __future__ import annotations

import html
from fnmatch import fnmatch
from typing import Any

import numpy as np

from brian2.core.preferences import prefs
from brian2.utils.logger import get_logger

logger = get_logger(__name__)

# --- Type aliases ---
ParamTuple = tuple[str, str, str]


def get_introspector() -> CppyyIntrospector | None:
    """
    Get the global introspector instance, or None if introspection is disabled.

    Returns None (not an error) when the preference is off, so callers can do:
        if intro := get_introspector():
            intro.list_objects()
    """
    return CppyyIntrospector.get_instance()


class CppyyIntrospector:
    """
    A live debugging interface into cppyy's JIT-compiled C++ code.

    This is a singleton — all code objects register with the same instance.
    Created lazily on first access when introspection is enabled.
    """

    _instance: CppyyIntrospector | None = None

    def __init__(self) -> None:
        # All registered code objects, keyed by name
        self._objects: dict[str, Any] = {}  # name → CppyyCodeObject

        # C++ source for each (codeobj_name, block) pair
        self._sources: dict[tuple[str, str], str] = {}

        # Original source and function ref, for restore()
        self._original_sources: dict[tuple[str, str], str] = {}
        self._original_funcs: dict[tuple[str, str], Any] = {}

        # Version counter for function replacement (can't redefine extern "C")
        self._version_counter: dict[tuple[str, str], int] = {}

        # Counter for eval_cpp one-off functions
        self._eval_counter: int = 0

    @classmethod
    def get_instance(cls) -> CppyyIntrospector | None:
        """Get or create the singleton. Returns None if introspection is disabled."""
        if not prefs.codegen.runtime.cppyy.enable_introspection:
            return None
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton. Useful between test runs."""
        cls._instance = None

    def register(self, codeobj: Any, block: str, source: str) -> None:
        """
        Record a code object and the C++ source it compiled.

        Called automatically by compile_block() when introspection is enabled.
        Stores both the code object reference (for live namespace access) and
        the source string (for display and function replacement).
        """
        name: str = codeobj.name
        self._objects[name] = codeobj
        self._sources[(name, block)] = source
        self._original_sources[(name, block)] = source
        self._original_funcs[(name, block)] = codeobj.compiled_code.get(block)
        logger.diagnostic(f"introspector: registered {name}.{block}")

    def _resolve_name(self, pattern: str) -> str:
        """
        Resolve a name or glob pattern to a single code object name.

        Allows shorthand like "stateupdater*" instead of the full
        "neurongroup_stateupdater_codeobject" name.
        """
        # Exact match first
        if pattern in self._objects:
            return pattern

        # Glob match
        matches: list[str] = [name for name in self._objects if fnmatch(name, pattern)]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) == 0:
            available: str = ", ".join(sorted(self._objects.keys()))
            raise KeyError(
                f"No code object matching '{pattern}'. Available: {available}"
            )
        else:
            raise KeyError(
                f"Pattern '{pattern}' matches multiple objects: {matches}. "
                f"Be more specific."
            )

    def _resolve_names(self, pattern: str) -> list[str]:
        """Resolve a pattern to all matching names (for list_objects filtering)."""
        if pattern == "*":
            return sorted(self._objects.keys())
        return sorted(name for name in self._objects if fnmatch(name, pattern))

    def list_objects(self, pattern: str = "*") -> ObjectListDisplay:
        """
        List all registered code objects, their blocks, and template types.

        Returns a display object that renders as a table in Jupyter or
        as formatted text in a terminal.
        """
        rows: list[dict[str, str]] = []
        for name in self._resolve_names(pattern):
            codeobj = self._objects[name]
            blocks: list[str] = [
                block
                for block in ("before_run", "run", "after_run")
                if (name, block) in self._sources
            ]
            rows.append(
                {
                    "name": name,
                    "template": getattr(codeobj, "template_name", "?"),
                    "blocks": ", ".join(blocks),
                    "num_vars": str(len(codeobj.variables)),
                }
            )
        return ObjectListDisplay(rows)

    def source(self, pattern: str, block: str = "run") -> SourceDisplay:
        """
        View the C++ source for a code object's block.

        Returns a display object with the source code. In Jupyter, this
        renders with basic syntax highlighting.
        """
        name: str = self._resolve_name(pattern)
        key: tuple[str, str] = (name, block)
        if key not in self._sources:
            available_blocks: list[str] = [b for n, b in self._sources if n == name]
            raise KeyError(
                f"No source for {name}.{block}. Available blocks: {available_blocks}"
            )
        return SourceDisplay(self._sources[key], title=f"{name}.{block}")

    def params(self, pattern: str, block: str = "run") -> ParamsDisplay:
        """
        View the parameter mapping for a code object's block.

        Shows how each C++ function parameter maps to a namespace key
        and its current runtime value.
        """
        name: str = self._resolve_name(pattern)
        codeobj = self._objects[name]
        mapping: list[ParamTuple] = codeobj._param_mappings.get(block, [])

        rows: list[dict[str, Any]] = []
        for i, (cpp_name, ns_key, c_type) in enumerate(mapping):
            val: Any = codeobj.namespace.get(ns_key, "<MISSING>")
            rows.append(
                {
                    "index": i,
                    "c_type": c_type,
                    "cpp_name": cpp_name,
                    "ns_key": ns_key,
                    "value": _describe_value(val),
                }
            )
        return ParamsDisplay(rows, title=f"{name}.{block}")

    def namespace(self, pattern: str) -> NamespaceDisplay:
        """
        View the full namespace dict for a code object, categorized by type.

        Categories: arrays, sizes, constants, variables, dynamic arrays, other.
        """
        name: str = self._resolve_name(pattern)
        codeobj = self._objects[name]
        ns: dict[str, Any] = codeobj.namespace

        categorized: dict[str, list[tuple[str, str]]] = {
            "arrays": [],
            "sizes": [],
            "constants": [],
            "variable_objects": [],
            "dynamic_arrays": [],
            "other": [],
        }

        for key in sorted(ns.keys()):
            val = ns[key]
            desc = _describe_value(val)

            if key.startswith("_ptr_array_"):
                categorized["arrays"].append((key, desc))
            elif key.startswith("_num"):
                categorized["sizes"].append((key, desc))
            elif key.startswith("_var_"):
                categorized["variable_objects"].append((key, desc))
            elif key.startswith("_dynamic_array_"):
                categorized["dynamic_arrays"].append((key, desc))
            elif isinstance(val, (int, float, np.integer, np.floating)):
                categorized["constants"].append((key, desc))
            else:
                categorized["other"].append((key, desc))

        return NamespaceDisplay(categorized, title=name)

    def inspect(self, pattern: str, block: str = "run") -> InspectDisplay:
        """
        Full inspection: source + params + namespace in one view.
        In Jupyter, renders as collapsible sections.
        """
        name: str = self._resolve_name(pattern)
        return InspectDisplay(
            source=self.source(name, block),
            params=self.params(name, block),
            namespace=self.namespace(name),
            title=f"{name}.{block}",
        )

    def cpp_globals(self) -> list[str]:
        """List all Brian-related symbols in cppyy's global namespace."""
        from .cppyy_rt import _get_cppyy

        cppyy = _get_cppyy()
        return sorted(x for x in dir(cppyy.gbl) if "_brian_" in x)

    def get_body(self, pattern: str, block: str = "run") -> str:
        """
        Extract just the function body from the compiled source.

        Returns the code between the outer { } of the function definition,
        ready for editing. Pass the modified body to replace_body().
        """
        name: str = self._resolve_name(pattern)
        source: str = self._sources[(name, block)]
        func_name: str = self._get_func_name(name, block)
        _, _, body = _extract_function_parts(source, func_name)
        return body

    def replace_body(self, pattern: str, block: str, new_body: str) -> str:
        """
        Replace a function's body while keeping its signature intact.

        Compiles the new body under a versioned function name (because Cling
        can't redefine extern "C" symbols), then swaps the code object's
        function reference so the next timestep uses the new version.

        The support code (e.g. _timestep inline) is already in Cling from the
        original compilation, so new_body can reference those functions freely.

        Returns the versioned function name for reference.
        """
        name: str = self._resolve_name(pattern)
        codeobj = self._objects[name]
        from .cppyy_rt import _get_cppyy

        cppyy = _get_cppyy()

        # Bump version counter — each replacement gets a unique name
        version: int = self._version_counter.get((name, block), 0) + 1
        self._version_counter[(name, block)] = version

        # Build the function signature from the param mapping
        mapping: list[ParamTuple] = codeobj._param_mappings[block]
        params_str: str = ", ".join(
            f"{c_type} {cpp_name}" for cpp_name, _, c_type in mapping
        )

        # Compile under a versioned name
        original_func_name: str = self._get_func_name(name, block)
        versioned_name: str = f"{original_func_name}_v{version}"

        new_source: str = (
            f'extern "C" void {versioned_name}({params_str}) {{\n{new_body}\n}}\n'
        )

        logger.info(
            f"introspector: compiling {versioned_name} (replacing {original_func_name})"
        )
        cppyy.cppdef(new_source)

        # Swap the function reference — next run_block() call uses the new one
        new_func: Any = getattr(cppyy.gbl, versioned_name)
        codeobj.compiled_code[block] = new_func

        # Track the replacement source (with the original function name for display)
        display_source: str = new_source.replace(versioned_name, original_func_name)
        self._sources[(name, block)] = display_source

        return versioned_name

    def replace_source(self, pattern: str, block: str, new_source: str) -> str:
        """
        Replace a function with completely new C++ source.

        For advanced users who need to modify support code or add new helpers.
        The function name in new_source is automatically versioned to avoid
        Cling redefinition errors.

        Warning: if new_source includes support code that's already defined
        in Cling (like _timestep), you'll get a redefinition error. Use
        inject_cpp() to add new helpers first, then replace_body() to use them.
        """
        name: str = self._resolve_name(pattern)
        codeobj = self._objects[name]
        from .cppyy_rt import _get_cppyy

        cppyy = _get_cppyy()

        version: int = self._version_counter.get((name, block), 0) + 1
        self._version_counter[(name, block)] = version

        original_func_name: str = self._get_func_name(name, block)
        versioned_name: str = f"{original_func_name}_v{version}"

        # Replace the function name in the user's source
        patched_source: str = new_source.replace(original_func_name, versioned_name)

        cppyy.cppdef(patched_source)
        new_func: Any = getattr(cppyy.gbl, versioned_name)
        codeobj.compiled_code[block] = new_func

        self._sources[(name, block)] = new_source
        self._version_counter[(name, block)] = version

        return versioned_name

    def restore(self, pattern: str, block: str = "run") -> None:
        """Restore the original compiled function, undoing any replace_body() calls."""
        name: str = self._resolve_name(pattern)
        key: tuple[str, str] = (name, block)

        if key not in self._original_funcs:
            raise KeyError(f"No original function stored for {name}.{block}")

        codeobj = self._objects[name]
        codeobj.compiled_code[block] = self._original_funcs[key]
        self._sources[key] = self._original_sources[key]
        self._version_counter.pop(key, None)

        logger.info(f"introspector: restored original {name}.{block}")

    def inject_cpp(self, code: str) -> None:
        """
        Compile arbitrary C++ code into Cling's interpreter.

        Use this to define helper functions, structs, or globals that your
        replacement function bodies can reference. For example:

            intro.inject_cpp('''
                inline double my_custom_activation(double x) {
                    return x > 0 ? x : 0.01 * x;  // leaky relu
                }
            ''')
        """
        from .cppyy_rt import _get_cppyy

        cppyy = _get_cppyy()
        cppyy.cppdef(code)
        logger.info("introspector: injected custom C++ code")

    def eval_cpp(self, expression: str, result_type: str = "double") -> Any:
        """
        Evaluate a C++ expression and return the result to Python.

        Compiles a tiny one-off function, calls it, and returns the value.
        Useful for checking constants, testing expressions, or reading globals.

        Examples:
            intro.eval_cpp("M_PI")                          # → 3.14159...
            intro.eval_cpp("_brian_mod(7, 3)", "int32_t")   # → 1
            intro.eval_cpp("sizeof(double)", "size_t")      # → 8
        """
        from .cppyy_rt import _get_cppyy

        cppyy = _get_cppyy()

        func_name: str = f"_brian_eval_{self._eval_counter}"
        self._eval_counter += 1

        cppyy.cppdef(
            f"{result_type} {func_name}() {{ return ({result_type})({expression}); }}"
        )
        return getattr(cppyy.gbl, func_name)()

    def snapshot(self, pattern: str) -> dict[str, Any]:
        """
        Capture a snapshot of a code object's current state.

        Returns a plain dict with source, params, namespace values, and
        version info. Useful for before/after comparisons when testing
        function replacements.
        """
        name: str = self._resolve_name(pattern)
        codeobj = self._objects[name]

        array_snapshot: dict[str, Any] = {}
        for key, val in codeobj.namespace.items():
            if isinstance(val, np.ndarray):
                array_snapshot[key] = {
                    "shape": val.shape,
                    "dtype": str(val.dtype),
                    "min": float(val.min()) if val.size > 0 else None,
                    "max": float(val.max()) if val.size > 0 else None,
                    "mean": float(val.mean()) if val.size > 0 else None,
                }

        return {
            "name": name,
            "sources": {
                block: src for (n, block), src in self._sources.items() if n == name
            },
            "versions": {
                block: ver
                for (n, block), ver in self._version_counter.items()
                if n == name
            },
            "arrays": array_snapshot,
        }

    ### Internal helpers

    def _get_func_name(self, name: str, block: str) -> str:
        """Build the C++ function name matching _make_func_name in codeobject.py."""
        safe: str = name.replace(".", "_").replace("*", "").replace("-", "_")
        return f"_brian_cppyy_{block}_{safe}"

    def _repr_html_(self) -> str:
        """Display a summary table when the introspector itself is shown in Jupyter."""
        return self.list_objects()._repr_html_()


def _describe_value(val: Any) -> str:
    """One-line description of a namespace value."""
    if isinstance(val, np.ndarray):
        if val.size <= 4:
            return f"ndarray({val.shape}, {val.dtype}) = {val.tolist()}"
        return (
            f"ndarray({val.shape}, {val.dtype}) "
            f"range=[{val.min():.4g}, {val.max():.4g}]"
        )
    elif isinstance(val, (int, np.integer)):
        return f"int = {val}"
    elif isinstance(val, (float, np.floating)):
        return f"float = {val:.6g}"
    elif hasattr(val, "__class__"):
        return f"{val.__class__.__name__}"
    else:
        return repr(val)


def _extract_function_parts(source: str, func_name: str) -> tuple[str, str, str]:
    """
    Split C++ source into (preamble, signature, body).

    Finds the function by name, locates the opening brace, then uses
    brace-depth counting to find the matching close. Works reliably
    on our generated code (well-formed, no string literals containing braces).
    """
    marker: str = f"void {func_name}"
    func_start: int = source.find(marker)
    if func_start == -1:
        raise ValueError(
            f"Could not find function '{func_name}' in source. "
            f"Source starts with: {source[:200]}..."
        )

    preamble: str = source[:func_start].rstrip()

    brace_pos: int = source.find("{", func_start)
    if brace_pos == -1:
        raise ValueError(f"No opening brace found after '{func_name}'")

    signature: str = source[func_start:brace_pos].strip()

    # Match braces to find the function body
    depth: int = 0
    for i in range(brace_pos, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                body: str = source[brace_pos + 1 : i]
                return preamble, signature, body

    raise ValueError(f"Unmatched braces in function '{func_name}'")


# --- CSS used by all display classes ---
_DISPLAY_CSS: str = """
<style>
.brian-intro { font-family: 'Menlo', 'Monaco', 'Consolas', monospace; font-size: 13px; }
.brian-intro table { border-collapse: collapse; width: 100%; margin: 8px 0; }
.brian-intro th { background: #2d2d2d; color: #e6e6e6; padding: 6px 10px; text-align: left; }
.brian-intro td { padding: 5px 10px; border-bottom: 1px solid #eee; }
.brian-intro tr:hover td { background: #f5f5f5; }
.brian-intro pre { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px;
                   overflow-x: auto; line-height: 1.4; }
.brian-intro .kw { color: #569cd6; }
.brian-intro .type { color: #4ec9b0; }
.brian-intro .str { color: #ce9178; }
.brian-intro .num { color: #b5cea8; }
.brian-intro .cmt { color: #6a9955; }
.brian-intro .fn { color: #dcdcaa; }
.brian-intro h3 { margin: 12px 0 4px 0; color: #333; }
.brian-intro details { margin: 6px 0; }
.brian-intro summary { cursor: pointer; font-weight: bold; color: #333; padding: 4px 0; }
.brian-intro .missing { color: #e74c3c; font-weight: bold; }
</style>
"""


def _highlight_cpp(source: str) -> str:
    """Basic C++ syntax highlighting for HTML display."""
    import re

    # Escape HTML first
    s: str = html.escape(source)

    # Comments (// to end of line)
    s = re.sub(r"(//.*?)$", r'<span class="cmt">\1</span>', s, flags=re.MULTILINE)

    # Keywords
    keywords = (
        r"\b(extern|void|const|for|if|else|return|static|inline|template|"
        r"typename|struct|namespace|typedef|using|auto|break|continue|"
        r"while|do|switch|case|default)\b"
    )
    s = re.sub(keywords, r'<span class="kw">\1</span>', s)

    # Types
    types = (
        r"\b(int|int8_t|int32_t|int64_t|size_t|long|double|float|char|"
        r"bool|unsigned|void)\b"
    )
    s = re.sub(types, r'<span class="type">\1</span>', s)

    # Numbers
    s = re.sub(
        r"\b(\d+\.?\d*(?:[eE][+-]?\d+)?[fFuUlL]*)\b",
        r'<span class="num">\1</span>',
        s,
    )

    return s


class ObjectListDisplay:
    """Display for list_objects() — table of registered code objects."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows: list[dict[str, str]] = rows

    def _repr_html_(self) -> str:
        header: str = (
            "<tr><th>Code Object</th><th>Template</th>"
            "<th>Compiled Blocks</th><th># Variables</th></tr>"
        )
        body: str = ""
        for row in self.rows:
            body += (
                f"<tr><td><code>{html.escape(row['name'])}</code></td>"
                f"<td>{html.escape(row['template'])}</td>"
                f"<td>{html.escape(row['blocks'])}</td>"
                f"<td>{html.escape(row['num_vars'])}</td></tr>"
            )
        return (
            f'{_DISPLAY_CSS}<div class="brian-intro">'
            f"<h3>Compiled Code Objects</h3>"
            f"<table>{header}{body}</table></div>"
        )

    def __repr__(self) -> str:
        lines: list[str] = ["Compiled Code Objects:", ""]
        for row in self.rows:
            lines.append(
                f"  {row['name']:<50s} template={row['template']:<20s} "
                f"blocks=[{row['blocks']}]  vars={row['num_vars']}"
            )
        return "\n".join(lines)


class SourceDisplay:
    """Display for source() — C++ code with highlighting."""

    def __init__(self, source: str, title: str = "") -> None:
        self.source: str = source
        self.title: str = title

    def _repr_html_(self) -> str:
        highlighted: str = _highlight_cpp(self.source)
        return (
            f'{_DISPLAY_CSS}<div class="brian-intro">'
            f"<h3>{html.escape(self.title)}</h3>"
            f"<pre>{highlighted}</pre></div>"
        )

    def __repr__(self) -> str:
        return f"--- {self.title} ---\n{self.source}"

    def __str__(self) -> str:
        return self.source


class ParamsDisplay:
    """Display for params() — parameter mapping table."""

    def __init__(self, rows: list[dict[str, Any]], title: str = "") -> None:
        self.rows: list[dict[str, Any]] = rows
        self.title: str = title

    def _repr_html_(self) -> str:
        header: str = (
            "<tr><th>#</th><th>C++ Type</th><th>Parameter Name</th>"
            "<th>Namespace Key</th><th>Current Value</th></tr>"
        )
        body: str = ""
        for row in self.rows:
            missing_cls: str = ' class="missing"' if "MISSING" in row["value"] else ""
            body += (
                f"<tr><td>{row['index']}</td>"
                f"<td><code>{html.escape(row['c_type'])}</code></td>"
                f"<td><code>{html.escape(row['cpp_name'])}</code></td>"
                f"<td><code>{html.escape(row['ns_key'])}</code></td>"
                f"<td{missing_cls}>{html.escape(row['value'])}</td></tr>"
            )
        return (
            f'{_DISPLAY_CSS}<div class="brian-intro">'
            f"<h3>Parameter Mapping: {html.escape(self.title)}</h3>"
            f"<table>{header}{body}</table></div>"
        )

    def __repr__(self) -> str:
        lines: list[str] = [f"Parameter Mapping: {self.title}", ""]
        for row in self.rows:
            lines.append(
                f"  [{row['index']:>2d}] {row['c_type']:<12s} "
                f"{row['cpp_name']:<44s} <- ns[{row['ns_key']}] = {row['value']}"
            )
        return "\n".join(lines)


class NamespaceDisplay:
    """Display for namespace() — categorized namespace contents."""

    def __init__(
        self,
        categorized: dict[str, list[tuple[str, str]]],
        title: str = "",
    ) -> None:
        self.categorized: dict[str, list[tuple[str, str]]] = categorized
        self.title: str = title

    # Friendly category labels
    _LABELS: dict[str, str] = {
        "arrays": "Arrays (data pointers)",
        "sizes": "Sizes (_num*)",
        "constants": "Constants (scalars)",
        "variable_objects": "Variable Objects (_var_*)",
        "dynamic_arrays": "Dynamic Arrays",
        "other": "Other",
    }

    def _repr_html_(self) -> str:
        sections: str = ""
        for cat, entries in self.categorized.items():
            if not entries:
                continue
            label: str = self._LABELS.get(cat, cat)
            rows: str = ""
            for key, desc in entries:
                rows += (
                    f"<tr><td><code>{html.escape(key)}</code></td>"
                    f"<td>{html.escape(desc)}</td></tr>"
                )
            sections += (
                f"<details open><summary>{html.escape(label)} "
                f"({len(entries)})</summary>"
                f"<table><tr><th>Key</th><th>Value</th></tr>"
                f"{rows}</table></details>"
            )
        return (
            f'{_DISPLAY_CSS}<div class="brian-intro">'
            f"<h3>Namespace: {html.escape(self.title)}</h3>"
            f"{sections}</div>"
        )

    def __repr__(self) -> str:
        lines: list[str] = [f"Namespace: {self.title}", ""]
        for cat, entries in self.categorized.items():
            if not entries:
                continue
            label: str = self._LABELS.get(cat, cat)
            lines.append(f"  [{label}]")
            for key, desc in entries:
                lines.append(f"    {key:<50s} {desc}")
            lines.append("")
        return "\n".join(lines)


class InspectDisplay:
    """Display for inspect() — combined source + params + namespace."""

    def __init__(
        self,
        source: SourceDisplay,
        params: ParamsDisplay,
        namespace: NamespaceDisplay,
        title: str = "",
    ) -> None:
        self.source: SourceDisplay = source
        self.params: ParamsDisplay = params
        self.namespace: NamespaceDisplay = namespace
        self.title: str = title

    def _repr_html_(self) -> str:
        return (
            f'{_DISPLAY_CSS}<div class="brian-intro">'
            f"<h2>Inspect: {html.escape(self.title)}</h2>"
            f"<details open><summary>C++ Source</summary>"
            f"<pre>{_highlight_cpp(self.source.source)}</pre></details>"
            f"<details open><summary>Parameter Mapping</summary>"
            f"{self.params._repr_html_()}</details>"
            f"<details><summary>Namespace (click to expand)</summary>"
            f"{self.namespace._repr_html_()}</details>"
            f"</div>"
        )

    def __repr__(self) -> str:
        return (
            f"{'=' * 60}\n"
            f"INSPECT: {self.title}\n"
            f"{'=' * 60}\n\n"
            f"{repr(self.source)}\n\n"
            f"{repr(self.params)}\n\n"
            f"{repr(self.namespace)}"
        )
