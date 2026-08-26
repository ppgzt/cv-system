#!/usr/bin/env python3
"""Valida e resume o microbenchmark instrumentado dos agentes PADE."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


COMPONENTS = ("visual", "selection", "preprocessing", "prediction")
METRICS = ("service_time_ms", "total_latency_ms")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--input-dir", required=True)
    return result


def stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(values.size), "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)), "std_sample_ms": float(values.std(ddof=1)),
        "min_ms": float(values.min()), "max_ms": float(values.max()),
        "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)),
    }


def main() -> None:
    root = Path(parser().parse_args().input_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected_n = int(metadata["measured_iterations_per_component"])
    if metadata["warmup_excluded"] != 50 or expected_n != 1000:
        raise ValueError("final Section 4.1 protocol requires warmup=50 and iterations=1000")
    summary: dict[str, dict] = {"metadata": metadata, "components": {}}
    lines = [
        "# Microbenchmark Instrumentado dos Agentes PADE", "",
        "A série principal é `service_time_ms`: somente a execução da função computacional no worker.",
        "`total_latency_ms` é secundária e inclui ACL local, OrderedInbox, espera de thread e callback do reactor.",
        "", "## Validação", "",
        f"- Warm-ups excluídos: {metadata['warmup_excluded']}",
        f"- Medições válidas por componente: {expected_n}",
        f"- Seed: {metadata['seed']}",
        f"- Selector: `{metadata['selector_model']}`", f"- Predictor: `{metadata['predictor_model']}`", "",
    ]
    for component in COMPONENTS:
        path = root / f"{component}_measurements.csv"
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != expected_n:
            raise ValueError(f"{component}: expected {expected_n} rows, got {len(rows)}")
        iterations = [int(row["iteration"]) for row in rows]
        if iterations != list(range(expected_n)):
            raise ValueError(f"{component}: iterations are not exactly 0..{expected_n - 1}")
        component_summary: dict[str, object] = {"gc_measurements": sum(row["gc_occurred"] == "true" for row in rows)}
        for metric in METRICS:
            values = np.array([float(row[metric]) for row in rows], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{component}: {metric} contains NaN or infinity")
            component_summary[metric] = stats(values)
        summary["components"][component] = component_summary

    lines += ["## Serviço computacional (métrica principal)", "", "| Componente | n | Média (ms) | Mediana | DP amostral | Mínimo | P95 | P99 | Máximo | GC em medições |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for component in COMPONENTS:
        result = summary["components"][component]
        value = result["service_time_ms"]
        lines.append(f"| {component} | {value['n']} | {value['mean_ms']:.4f} | {value['median_ms']:.4f} | {value['std_sample_ms']:.4f} | {value['min_ms']:.4f} | {value['p95_ms']:.4f} | {value['p99_ms']:.4f} | {value['max_ms']:.4f} | {result['gc_measurements']} |")
    lines += ["", "## Latência total local (métrica secundária)", "", "| Componente | n | Média (ms) | Mediana | DP amostral | Mínimo | P95 | P99 | Máximo |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for component in COMPONENTS:
        value = summary["components"][component]["total_latency_ms"]
        lines.append(f"| {component} | {value['n']} | {value['mean_ms']:.4f} | {value['median_ms']:.4f} | {value['std_sample_ms']:.4f} | {value['min_ms']:.4f} | {value['p95_ms']:.4f} | {value['p99_ms']:.4f} | {value['max_ms']:.4f} |")
    lines += ["", "## Sanity checks", "", "- Exatamente 1.000 linhas `iteration=0..999` por componente: PASS.", "- Warm-ups não aparecem nos CSVs: PASS.", "- Valores finitos para `service_time_ms` e `total_latency_ms`: PASS.", "- Configuração de modelos/preprocessamentos registrada em `metadata.json`: PASS.", "- Nenhum CSV de benchmark histórico ou IEEE foi lido: PASS.", ""]
    for component in COMPONENTS:
        value = summary["components"][component]["service_time_ms"]
        if value["max_ms"] > 5 * value["median_ms"]:
            lines += ["## Sinalização de pausa extrema", "", f"`{component}` possui máximo de serviço acima de 5× a mediana. O valor foi preservado; investigar GC/infraestrutura antes de criar uma série alternativa.", ""]
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"__ANALYSIS_REPORT__={root / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
