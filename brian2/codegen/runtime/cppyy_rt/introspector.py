"""
Runtime introspection for the cppyy backend.

Enable with: prefs.codegen.runtime.cppyy.enable_introspection = True

Usage:
    from brian2.codegen.runtime.cppyy_rt.introspector import get_introspector
    intro = get_introspector()

    intro.list_objects()                          # see all compiled code objects
    intro.source("*stateupdater*")                # view generated C++
    intro.params("*stateupdater*")                # view parameter mapping
    intro.namespace("*stateupdater*")             # view runtime values
    intro.inspect("*stateupdater*")               # all of the above

    body = intro.get_body("*stateupdater*", "run")
    new_body = body.replace("exp(_lio_2)", "1.0 + _lio_2")
    intro.replace_body("*stateupdater*", "run", new_body)
    intro.restore("*stateupdater*", "run")        # undo
"""

from __future__ import annotations

import html
import re as _re
from fnmatch import fnmatch
from typing import Any

import numpy as np

from brian2.core.preferences import prefs
from brian2.utils.logger import get_logger

logger = get_logger(__name__)

ParamTuple = tuple[str, str, str]

# --- Optional rich support ---
_RICH_AVAILABLE: bool = False
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree

    _RICH_AVAILABLE = True
except ImportError:
    pass


def get_introspector() -> CppyyIntrospector | None:
    """Get the global introspector, or None if disabled."""
    return CppyyIntrospector.get_instance()


class CppyyIntrospector:
    """
    Live debugging interface into cppyy's JIT-compiled C++ code.

    Singleton — all code objects register with the same instance.
    """

    _instance: CppyyIntrospector | None = None

    def __init__(self) -> None:
        self._objects: dict[str, Any] = {}
        self._sources: dict[tuple[str, str], str] = {}
        self._original_sources: dict[tuple[str, str], str] = {}
        self._original_funcs: dict[tuple[str, str], Any] = {}
        self._version_counter: dict[tuple[str, str], int] = {}
        self._eval_counter: int = 0
        # Track registration order so we can prefer latest
        self._registration_order: list[str] = []

    @classmethod
    def get_instance(cls) -> CppyyIntrospector | None:
        if not prefs.codegen.runtime.cppyy.enable_introspection:
            return None
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    # === Registration ===

    def register(self, codeobj: Any, block: str, source: str) -> None:
        name: str = codeobj.name
        self._objects[name] = codeobj
        self._sources[(name, block)] = source
        self._original_sources[(name, block)] = source
        self._original_funcs[(name, block)] = codeobj.compiled_code.get(block)
        if name not in self._registration_order:
            self._registration_order.append(name)
        logger.diagnostic(f"introspector: registered {name}.{block}")

    # === Name resolution ===

    def _resolve_name(self, pattern: str) -> str:
        """
        Resolve a name or glob pattern to a single code object name.

        When multiple objects match (e.g. *stateupdater* matching both
        _codeobject and _codeobject_1), we prefer the LATEST registered
        match. This is usually what the user wants — after a second run(),
        the new code object is the active one. But if the user modified
        the original and is calling restore(), the original is still there.

        If disambiguation is needed, uses these heuristics:
          1. If one match lacks a trailing _\d+ suffix and others have one,
             prefer the base name (the "original" code object).
          2. Otherwise prefer the most recently registered.
        """
        if pattern in self._objects:
            return pattern

        matches: list[str] = [name for name in self._objects if fnmatch(name, pattern)]

        if len(matches) == 0:
            available: str = ", ".join(sorted(self._objects.keys()))
            raise KeyError(
                f"No code object matching '{pattern}'. Available: {available}"
            )

        if len(matches) == 1:
            return matches[0]

        # Multiple matches — try to pick the most useful one.
        # Prefer the base name (without _1, _2 suffix) if it exists.
        base_matches: list[str] = [m for m in matches if not _re.search(r"_\d+$", m)]
        if len(base_matches) == 1:
            return base_matches[0]

        # Fall back to most recently registered
        for name in reversed(self._registration_order):
            if name in matches:
                return name

        return matches[0]

    def _resolve_names(self, pattern: str) -> list[str]:
        if pattern == "*":
            return sorted(self._objects.keys())
        return sorted(name for name in self._objects if fnmatch(name, pattern))

    # === Inspection ===

    def list_objects(self, pattern: str = "*") -> ObjectListDisplay:
        """List all registered code objects, their blocks, and template types."""
        rows: list[dict[str, str]] = []
        for name in self._resolve_names(pattern):
            codeobj = self._objects[name]
            blocks: list[str] = [
                block
                for block in ("before_run", "run", "after_run")
                if (name, block) in self._sources
            ]
            is_active: bool = name in self._registration_order[-len(self._objects) :]
            rows.append(
                {
                    "name": name,
                    "template": getattr(codeobj, "template_name", "?"),
                    "blocks": ", ".join(blocks),
                    "num_vars": str(len(codeobj.variables)),
                    "active": "●" if is_active else "○",
                }
            )
        return ObjectListDisplay(rows)

    def source(self, pattern: str, block: str = "run") -> SourceDisplay:
        """View the C++ source for a code object's block."""
        name: str = self._resolve_name(pattern)
        key: tuple[str, str] = (name, block)
        if key not in self._sources:
            available_blocks: list[str] = [b for n, b in self._sources if n == name]
            raise KeyError(
                f"No source for {name}.{block}. Available blocks: {available_blocks}"
            )
        return SourceDisplay(self._sources[key], title=f"{name}.{block}")

    def params(self, pattern: str, block: str = "run") -> ParamsDisplay:
        """View parameter mapping with current runtime values."""
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
        """View the full namespace dict, categorized by type."""
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
        """Full inspection: source + params + namespace in one view."""
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

    # === Modification ===

    def get_body(self, pattern: str, block: str = "run") -> str:
        """Extract just the function body, ready for editing."""
        name: str = self._resolve_name(pattern)
        source: str = self._sources[(name, block)]
        func_name: str = self._get_func_name(name, block)
        _, _, body = _extract_function_parts(source, func_name)
        return body

    def replace_body(self, pattern: str, block: str, new_body: str) -> str:
        """
        Replace a function's body, keeping its signature.

        Compiles under a versioned name (_v1, _v2...) since Cling can't
        redefine extern "C" symbols. Swaps the code object's function ref.
        Returns the versioned function name.
        """
        name: str = self._resolve_name(pattern)
        codeobj = self._objects[name]
        from .cppyy_rt import _get_cppyy

        cppyy = _get_cppyy()

        version: int = self._version_counter.get((name, block), 0) + 1
        self._version_counter[(name, block)] = version

        mapping: list[ParamTuple] = codeobj._param_mappings[block]
        params_str: str = ", ".join(
            f"{c_type} {cpp_name}" for cpp_name, _, c_type in mapping
        )

        original_func_name: str = self._get_func_name(name, block)
        versioned_name: str = f"{original_func_name}_v{version}"

        new_source: str = (
            f'extern "C" void {versioned_name}({params_str}) {{\n{new_body}\n}}\n'
        )

        logger.info(
            f"introspector: compiling {versioned_name} (replacing {original_func_name})"
        )
        cppyy.cppdef(new_source)

        new_func: Any = getattr(cppyy.gbl, versioned_name)
        codeobj.compiled_code[block] = new_func

        display_source: str = new_source.replace(versioned_name, original_func_name)
        self._sources[(name, block)] = display_source

        return versioned_name

    def replace_source(self, pattern: str, block: str, new_source: str) -> str:
        """Replace with completely new C++ source. Function name auto-versioned."""
        name: str = self._resolve_name(pattern)
        codeobj = self._objects[name]
        from .cppyy_rt import _get_cppyy

        cppyy = _get_cppyy()

        version: int = self._version_counter.get((name, block), 0) + 1
        self._version_counter[(name, block)] = version

        original_func_name: str = self._get_func_name(name, block)
        versioned_name: str = f"{original_func_name}_v{version}"

        patched_source: str = new_source.replace(original_func_name, versioned_name)
        cppyy.cppdef(patched_source)
        new_func: Any = getattr(cppyy.gbl, versioned_name)
        codeobj.compiled_code[block] = new_func

        self._sources[(name, block)] = new_source
        return versioned_name

    def restore(self, pattern: str, block: str = "run") -> None:
        """Restore the original compiled function."""
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
        """Compile arbitrary C++ into Cling (define helpers, structs, etc.)."""
        from .cppyy_rt import _get_cppyy

        cppyy = _get_cppyy()
        cppyy.cppdef(code)
        logger.info("introspector: injected custom C++ code")

    def eval_cpp(self, expression: str, result_type: str = "double") -> Any:
        """Evaluate a C++ expression and return the result."""
        from .cppyy_rt import _get_cppyy

        cppyy = _get_cppyy()
        func_name: str = f"_brian_eval_{self._eval_counter}"
        self._eval_counter += 1
        cppyy.cppdef(
            f"{result_type} {func_name}() {{ return ({result_type})({expression}); }}"
        )
        return getattr(cppyy.gbl, func_name)()

    def snapshot(self, pattern: str) -> dict[str, Any]:
        """Capture current state as a plain dict (for comparisons)."""
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

    # === Rich CLI display ===

    def print_objects(self, pattern: str = "*") -> None:
        """Pretty-print all code objects to the terminal."""
        display = self.list_objects(pattern)
        if _RICH_AVAILABLE:
            _rich_print_objects(display)
        else:
            print(repr(display))

    def print_source(self, pattern: str, block: str = "run") -> None:
        """Pretty-print C++ source with syntax highlighting."""
        display = self.source(pattern, block)
        if _RICH_AVAILABLE:
            _rich_print_source(display)
        else:
            print(repr(display))

    def print_params(self, pattern: str, block: str = "run") -> None:
        """Pretty-print parameter mapping."""
        display = self.params(pattern, block)
        if _RICH_AVAILABLE:
            _rich_print_params(display)
        else:
            print(repr(display))

    def print_namespace(self, pattern: str) -> None:
        """Pretty-print namespace contents."""
        display = self.namespace(pattern)
        if _RICH_AVAILABLE:
            _rich_print_namespace(display)
        else:
            print(repr(display))

    def print_inspect(self, pattern: str, block: str = "run") -> None:
        """Pretty-print full inspection (source + params + namespace)."""
        display = self.inspect(pattern, block)
        if _RICH_AVAILABLE:
            _rich_print_inspect(display)
        else:
            print(repr(display))

    # === Internal ===

    def _get_func_name(self, name: str, block: str) -> str:
        safe: str = name.replace(".", "_").replace("*", "").replace("-", "_")
        return f"_brian_cppyy_{block}_{safe}"

    def _repr_html_(self) -> str:
        return self.list_objects()._repr_html_()


# =========================================================================
# Rich CLI renderers (only used when `rich` is installed)
# =========================================================================


def _rich_print_objects(display: ObjectListDisplay) -> None:
    console = Console()
    table = Table(
        title="[bold]Compiled Code Objects[/bold]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("", width=1)
    table.add_column("Code Object", style="green")
    table.add_column("Template", style="yellow")
    table.add_column("Blocks")
    table.add_column("# Vars", justify="right")

    for row in display.rows:
        table.add_row(
            row.get("active", "●"),
            row["name"],
            row["template"],
            row["blocks"],
            row["num_vars"],
        )
    console.print(table)


def _rich_print_source(display: SourceDisplay) -> None:
    console = Console()
    syntax = Syntax(
        display.source, "cpp", theme="monokai", line_numbers=True, word_wrap=False
    )
    console.print(
        Panel(syntax, title=f"[bold]{display.title}[/bold]", border_style="cyan")
    )


def _rich_print_params(display: ParamsDisplay) -> None:
    console = Console()
    table = Table(
        title=f"[bold]Parameter Mapping: {display.title}[/bold]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("C++ Type", style="magenta")
    table.add_column("Parameter Name", style="green")
    table.add_column("Namespace Key", style="yellow")
    table.add_column("Current Value")

    for row in display.rows:
        val_style = "red bold" if "MISSING" in row["value"] else ""
        table.add_row(
            str(row["index"]),
            row["c_type"],
            row["cpp_name"],
            row["ns_key"],
            Text(row["value"], style=val_style),
        )
    console.print(table)


def _rich_print_namespace(display: NamespaceDisplay) -> None:
    console = Console()
    tree = Tree(f"[bold]Namespace: {display.title}[/bold]")

    labels: dict[str, str] = {
        "arrays": "[cyan]Arrays (data pointers)[/cyan]",
        "sizes": "[yellow]Sizes (_num*)[/yellow]",
        "constants": "[green]Constants (scalars)[/green]",
        "variable_objects": "[dim]Variable Objects (_var_*)[/dim]",
        "dynamic_arrays": "[magenta]Dynamic Arrays[/magenta]",
        "other": "[dim]Other[/dim]",
    }

    for cat, entries in display.categorized.items():
        if not entries:
            continue
        branch = tree.add(f"{labels.get(cat, cat)} ({len(entries)})")
        for key, desc in entries:
            branch.add(f"[bold]{key}[/bold]  →  {desc}")

    console.print(tree)


def _rich_print_inspect(display: InspectDisplay) -> None:
    console = Console()
    console.print()
    console.rule(f"[bold cyan]Inspect: {display.title}[/bold cyan]")
    console.print()

    _rich_print_source(display.source)
    console.print()
    _rich_print_params(display.params)
    console.print()
    _rich_print_namespace(display.namespace)


# =========================================================================
# Value description helper
# =========================================================================


def _describe_value(val: Any) -> str:
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
        return val.__class__.__name__
    else:
        return repr(val)


def _extract_function_parts(source: str, func_name: str) -> tuple[str, str, str]:
    """Split C++ source into (preamble, signature, body) by brace matching."""
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


# =========================================================================
# Jupyter HTML display classes
# =========================================================================

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
.brian-intro h3 { margin: 12px 0 4px 0; color: #333; }
.brian-intro details { margin: 6px 0; }
.brian-intro summary { cursor: pointer; font-weight: bold; color: #333; padding: 4px 0; }
.brian-intro .missing { color: #e74c3c; font-weight: bold; }
</style>
"""


def _highlight_cpp(source: str) -> str:
    s: str = html.escape(source)
    s = _re.sub(r"(//.*?)$", r'<span class="cmt">\1</span>', s, flags=_re.MULTILINE)
    s = _re.sub(
        r"\b(extern|void|const|for|if|else|return|static|inline|template|"
        r"typename|struct|namespace|typedef|using|auto|break|continue|"
        r"while|do|switch|case|default)\b",
        r'<span class="kw">\1</span>',
        s,
    )
    s = _re.sub(
        r"\b(int|int8_t|int32_t|int64_t|size_t|long|double|float|char|"
        r"bool|unsigned|void)\b",
        r'<span class="type">\1</span>',
        s,
    )
    s = _re.sub(
        r"\b(\d+\.?\d*(?:[eE][+-]?\d+)?[fFuUlL]*)\b",
        r'<span class="num">\1</span>',
        s,
    )
    return s


class ObjectListDisplay:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def _repr_html_(self) -> str:
        header = (
            "<tr><th>Code Object</th><th>Template</th>"
            "<th>Compiled Blocks</th><th># Variables</th></tr>"
        )
        body = ""
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
        lines = ["Compiled Code Objects:", ""]
        for row in self.rows:
            lines.append(
                f"  {row.get('active', '●')} {row['name']:<50s} "
                f"template={row['template']:<20s} "
                f"blocks=[{row['blocks']}]  vars={row['num_vars']}"
            )
        return "\n".join(lines)


class SourceDisplay:
    def __init__(self, source: str, title: str = "") -> None:
        self.source = source
        self.title = title

    def _repr_html_(self) -> str:
        return (
            f'{_DISPLAY_CSS}<div class="brian-intro">'
            f"<h3>{html.escape(self.title)}</h3>"
            f"<pre>{_highlight_cpp(self.source)}</pre></div>"
        )

    def __repr__(self) -> str:
        return f"--- {self.title} ---\n{self.source}"

    def __str__(self) -> str:
        return self.source


class ParamsDisplay:
    def __init__(self, rows: list[dict[str, Any]], title: str = "") -> None:
        self.rows = rows
        self.title = title

    def _repr_html_(self) -> str:
        header = (
            "<tr><th>#</th><th>C++ Type</th><th>Parameter Name</th>"
            "<th>Namespace Key</th><th>Current Value</th></tr>"
        )
        body = ""
        for row in self.rows:
            missing_cls = ' class="missing"' if "MISSING" in row["value"] else ""
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
        lines = [f"Parameter Mapping: {self.title}", ""]
        for row in self.rows:
            lines.append(
                f"  [{row['index']:>2d}] {row['c_type']:<12s} "
                f"{row['cpp_name']:<44s} <- ns[{row['ns_key']}] = {row['value']}"
            )
        return "\n".join(lines)


class NamespaceDisplay:
    _LABELS: dict[str, str] = {
        "arrays": "Arrays (data pointers)",
        "sizes": "Sizes (_num*)",
        "constants": "Constants (scalars)",
        "variable_objects": "Variable Objects (_var_*)",
        "dynamic_arrays": "Dynamic Arrays",
        "other": "Other",
    }

    def __init__(
        self, categorized: dict[str, list[tuple[str, str]]], title: str = ""
    ) -> None:
        self.categorized = categorized
        self.title = title

    def _repr_html_(self) -> str:
        sections = ""
        for cat, entries in self.categorized.items():
            if not entries:
                continue
            label = self._LABELS.get(cat, cat)
            rows = ""
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
        lines = [f"Namespace: {self.title}", ""]
        for cat, entries in self.categorized.items():
            if not entries:
                continue
            label = self._LABELS.get(cat, cat)
            lines.append(f"  [{label}]")
            for key, desc in entries:
                lines.append(f"    {key:<50s} {desc}")
            lines.append("")
        return "\n".join(lines)


class InspectDisplay:
    def __init__(
        self,
        source: SourceDisplay,
        params: ParamsDisplay,
        namespace: NamespaceDisplay,
        title: str = "",
    ) -> None:
        self.source = source
        self.params = params
        self.namespace = namespace
        self.title = title

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
