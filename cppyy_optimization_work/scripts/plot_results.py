"""Create summary figures from benchmark JSON results.

The benchmark harness writes one JSON file per run under raw_results/. This script reads
those files, aggregates repeated measurements with medians, and recreates the summary
figures used during cppyy optimization work.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw_results"
FIG_DIR = ROOT / "figures"

WORKLOAD_NAMES = {
    "small_lif": "Small LIF",
    "coba": "COBA",
    "kremer3": "Kremer3",
}

COLORS = {
    "cython": "#9a4343",
    "cppyy": "#239e91",
    "baseline": "#9da9b5",
    "after_A": "#5d8bc9",
    "after_AB": "#3f78bf",
    "after_ABC": "#2f70bf",
    "final": "#209e91",
    "v2_clean": "#209e91",
    "v2_combined": "#209e91",
    "v2_no_overload": "#72b6ad",
}


def load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    label = data.get("meta", {}).get("label") or path.stem
    return {
        "path": path,
        "label": label,
        "records": data.get("records", []),
    }


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    results = [load_result(path) for path in paths]
    return [result for result in results if result["records"]]


def median(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    return statistics.median(clean)


def workload_order(workloads: set[str]) -> list[str]:
    preferred = ["small_lif", "coba", "kremer3"]
    ordered = [name for name in preferred if name in workloads]
    ordered.extend(sorted(workloads - set(ordered)))
    return ordered


def records_for(
    results: list[dict[str, Any]],
    label: str,
    *,
    target: str | None = None,
    cold: bool | None = None,
    iter_value: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for result in results:
        if result["label"] != label:
            continue
        for record in result["records"]:
            if target is not None and record.get("target") != target:
                continue
            if cold is not None and record.get("cold") is not cold:
                continue
            if iter_value is not None and record.get("iter") != iter_value:
                continue
            out.append(record)
    return out


def medians_by_workload(records: list[dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        workload = record.get("workload")
        value = record.get(field)
        if workload and isinstance(value, (int, float)):
            grouped[workload].append(float(value))
    return {workload: value for workload, values in grouped.items() if (value := median(values))}


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    return plt, np


def label_for_workload(name: str) -> str:
    return WORKLOAD_NAMES.get(name, name)


def annotate_bars(ax: Any, bars: Any, suffix: str = "s") -> None:
    for bar in bars:
        height = bar.get_height()
        if height <= 0 or not math.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_cold_total(
    results: list[dict[str, Any]],
    output_dir: Path,
    *,
    cold_label: str,
) -> Path | None:
    plt, np = require_matplotlib()

    cython = medians_by_workload(
        records_for(results, cold_label, target="cython", cold=True),
        "total_s",
    )
    cppyy = medians_by_workload(
        records_for(results, cold_label, target="cppyy", cold=True),
        "total_s",
    )
    workloads = workload_order(set(cython) & set(cppyy))
    if not workloads:
        return None

    x = np.arange(len(workloads))
    width = 0.34
    fig, ax = plt.subplots(figsize=(12, 6.9))
    bars_cy = ax.bar(
        x - width / 2,
        [cython[w] for w in workloads],
        width,
        label="Cython cold total",
        color=COLORS["cython"],
    )
    bars_cp = ax.bar(
        x + width / 2,
        [cppyy[w] for w in workloads],
        width,
        label="cppyy cold total",
        color=COLORS["cppyy"],
    )
    annotate_bars(ax, bars_cy)
    annotate_bars(ax, bars_cp)

    for i, workload in enumerate(workloads):
        if cppyy[workload] > 0:
            ratio = cython[workload] / cppyy[workload]
            y = max(cython[workload], cppyy[workload]) * 1.07
            ax.text(i, y, f"{ratio:.1f}x faster", ha="center", fontsize=12, weight="bold")

    ax.set_title("Cold first-run total time", fontsize=18)
    ax.set_ylabel("Total time (seconds, lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for_workload(w) for w in workloads])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out = output_dir / "cold_total_comparison.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_warm_sim_progression(
    results: list[dict[str, Any]],
    output_dir: Path,
    *,
    labels: list[str],
    cython_label: str,
) -> Path | None:
    plt, np = require_matplotlib()

    series: list[tuple[str, dict[str, float], str]] = []
    display_names = {
        "baseline": "cppyy baseline",
        "after_A": "cppyy after A",
        "after_AB": "cppyy after AB",
        "after_ABC": "cppyy v1 fixes",
        "final": "cppyy final",
        "v2_clean": "cppyy current/v2",
        "v2_combined": "cppyy current/v2",
    }
    for label in labels:
        values = medians_by_workload(
            records_for(results, label, target="cppyy", cold=False),
            "sim_s",
        )
        if values:
            series.append((display_names.get(label, label), values, COLORS.get(label, "#4d79a8")))

    cython_values = medians_by_workload(
        records_for(results, cython_label, target="cython", cold=False),
        "sim_s",
    )
    if cython_values:
        series.append(("Cython warm", cython_values, "#222222"))

    workloads = workload_order(set.intersection(*(set(values) for _, values, _ in series))) if series else []
    if not workloads or not series:
        return None

    x = np.arange(len(workloads))
    width = min(0.8 / len(series), 0.22)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(12, 6.9))
    for offset, (name, values, color) in zip(offsets, series):
        bars = ax.bar(
            x + offset,
            [values[w] for w in workloads],
            width,
            label=name,
            color=color,
        )
        annotate_bars(ax, bars)

    ax.set_title("Warm simulation time progression", fontsize=18)
    ax.set_ylabel("Simulation time (seconds, lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for_workload(w) for w in workloads])
    ax.legend(ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out = output_dir / "warm_sim_progression.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_warm_ratio_progression(
    results: list[dict[str, Any]],
    output_dir: Path,
    *,
    labels: list[str],
    cython_label: str,
) -> Path | None:
    plt, np = require_matplotlib()

    cython = medians_by_workload(
        records_for(results, cython_label, target="cython", cold=False),
        "sim_s",
    )
    if not cython:
        return None

    display_names = {
        "baseline": "baseline",
        "after_A": "after A",
        "after_AB": "after AB",
        "after_ABC": "v1 fixes",
        "final": "final",
        "v2_clean": "current/v2",
        "v2_combined": "current/v2",
    }
    series: list[tuple[str, dict[str, float], str]] = []
    for label in labels:
        cppyy = medians_by_workload(
            records_for(results, label, target="cppyy", cold=False),
            "sim_s",
        )
        workloads = set(cython) & set(cppyy)
        ratios = {workload: cppyy[workload] / cython[workload] for workload in workloads}
        if ratios:
            series.append((display_names.get(label, label), ratios, COLORS.get(label, "#4d79a8")))

    workloads = workload_order(set.intersection(*(set(values) for _, values, _ in series))) if series else []
    if not workloads:
        return None

    x = np.arange(len(workloads))
    fig, ax = plt.subplots(figsize=(12, 6.9))
    for name, ratios, color in series:
        y = [ratios[w] for w in workloads]
        ax.plot(x, y, marker="o", linewidth=2.5, label=name, color=color)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.03, f"{yi:.2f}x", ha="center", va="bottom", fontsize=9)

    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.6, label="Cython parity")
    ax.set_title("cppyy / Cython warm simulation ratio", fontsize=18)
    ax.set_ylabel("Ratio (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for_workload(w) for w in workloads])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = output_dir / "warm_ratio_progression.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        records.append(json.loads(line))
    return records


def plot_multirun(
    output_dir: Path,
    *,
    cython_jsonl: Path | None,
    cppyy_jsonl: Path | None,
) -> Path | None:
    if cython_jsonl is None or cppyy_jsonl is None:
        return None

    cython_records = load_jsonl(cython_jsonl)
    cppyy_records = load_jsonl(cppyy_jsonl)
    if not cython_records or not cppyy_records:
        return None

    plt, np = require_matplotlib()

    cython_totals = [float(r["total_s"]) for r in cython_records]
    cppyy_totals = [float(r["total_s"]) for r in cppyy_records]
    n = min(len(cython_totals), len(cppyy_totals))
    cython_totals = cython_totals[:n]
    cppyy_totals = cppyy_totals[:n]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={"width_ratios": [1, 1.45]})
    ax = axes[0]
    total_cython = sum(cython_totals)
    total_cppyy = sum(cppyy_totals)
    bars = ax.bar(
        [0, 1],
        [total_cython, total_cppyy],
        color=[COLORS["cython"], COLORS["cppyy"]],
    )
    annotate_bars(ax, bars)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Cython\n{n} runs", f"cppyy\n{n} runs"])
    ax.set_ylabel("Total time (seconds)")
    ax.set_title("Repeated same-network runs")
    ax.grid(axis="y", alpha=0.25)
    if total_cppyy > 0:
        ax.text(
            0.5,
            max(total_cython, total_cppyy) * 0.82,
            f"{total_cython / total_cppyy:.1f}x faster\nover {n} runs",
            ha="center",
            fontsize=12,
            weight="bold",
        )

    ax = axes[1]
    x = np.arange(n)
    ax.plot(x, cython_totals, marker="o", linewidth=2.4, color=COLORS["cython"], label="Cython")
    ax.plot(x, cppyy_totals, marker="o", linewidth=2.4, color=COLORS["cppyy"], label="cppyy")
    ax.set_yscale("log")
    ax.set_title("Cold first iter plus warm repeated iters")
    ax.set_xlabel("Iteration in same process")
    ax.set_ylabel("Per-iteration total, log scale (seconds)")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend()

    fig.tight_layout()
    out = output_dir / "multirun_comparison.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def default_paths() -> list[Path]:
    return sorted(RAW_DIR.glob("*.json"))


def choose_default_labels(results: list[dict[str, Any]]) -> list[str]:
    available = {result["label"] for result in results}
    preferred = ["baseline", "after_ABC", "v2_clean"]
    labels = [label for label in preferred if label in available]
    if labels:
        return labels
    return sorted(label for label in available if label)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create matplotlib figures from cppyy benchmark JSON results."
    )
    parser.add_argument(
        "json_files",
        nargs="*",
        type=Path,
        help="Harness JSON files. Defaults to raw_results/*.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    parser.add_argument(
        "--progression-labels",
        nargs="+",
        default=None,
        help="Labels to compare for cppyy progression plots.",
    )
    parser.add_argument(
        "--cython-label",
        default=None,
        help="Label containing warm Cython records. Defaults to final, then baseline.",
    )
    parser.add_argument(
        "--cold-label",
        default=None,
        help="Label containing cold Cython and cppyy records. Defaults to cold, then baseline.",
    )
    parser.add_argument(
        "--multirun-cython",
        type=Path,
        default=None,
        help="Optional JSONL output from bench_multirun.py cython.",
    )
    parser.add_argument(
        "--multirun-cppyy",
        type=Path,
        default=None,
        help="Optional JSONL output from bench_multirun.py cppyy.",
    )
    args = parser.parse_args()

    paths = args.json_files or default_paths()
    results = load_results(paths)
    if not results:
        raise SystemExit("No benchmark records found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    available = {result["label"] for result in results}

    progression_labels = args.progression_labels or choose_default_labels(results)
    cython_label = args.cython_label
    if cython_label is None:
        cython_label = "final" if "final" in available else "baseline"
    cold_label = args.cold_label
    if cold_label is None:
        cold_label = "cold" if "cold" in available else "baseline"

    written: list[Path] = []
    for out in [
        plot_cold_total(results, args.output_dir, cold_label=cold_label),
        plot_warm_sim_progression(
            results,
            args.output_dir,
            labels=progression_labels,
            cython_label=cython_label,
        ),
        plot_warm_ratio_progression(
            results,
            args.output_dir,
            labels=progression_labels,
            cython_label=cython_label,
        ),
        plot_multirun(
            args.output_dir,
            cython_jsonl=args.multirun_cython,
            cppyy_jsonl=args.multirun_cppyy,
        ),
    ]:
        if out is not None:
            written.append(out)

    if not written:
        raise SystemExit("No figures were written. Check labels and input files.")

    print("Wrote figures:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
