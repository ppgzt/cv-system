#!/usr/bin/env python3
"""Ablacao offline e restrita de pre-processamentos para o PDI do Visual Event.

Nao altera runtime, agentes ou dados do dataset. O alvo e o mesmo das auditorias
PDI anteriores: ``parcial``/``suited`` sao positivos, ``background`` negativo e
``ruido`` e tratado pelo quality gate P99 existente.

As variantes sao deliberadamente pequenas: baseline, Gaussian 3/5, median 3/5,
morfologia 3x3, uma combinacao Gaussian+melhor morfologia e dois smoothers
causais do score. Nao ha busca de ROI, thresholds de pixel ou modelos aprendidos.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage, stats


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, REPO_ROOT / "data-analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import visual_event_classical_pdi_audit as classical  # noqa: E402
import visual_event_diagnostic as base  # noqa: E402
import visual_event_noise_pdi_audit as prior  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "data-analysis/visual_event_preprocessing_ablation/output"
DEFAULT_FRAME_FEATURES = (
    REPO_ROOT / "data-analysis/visual_event_noise_pdi_output/frame_features.csv"
)
EXPECTED_COHORT = (184, 13_741, 1_655)
ROI_B = (0.30, 0.70, 0.20, 0.80)
PIXEL_THRESHOLD_MM = 200.0
QUALITY_P99_THRESHOLD_MM = 2230.0
IDLE_PATIENCE = 3
BENCHMARK_PAIRS = 500

# A ordem e parte do protocolo: nao ha grid search alem destas configuracoes.
PHASE_ONE_VARIANTS = {
    "V0_baseline": {"family": "V0", "preprocessing": "none", "morphology": "none"},
    "V1_gaussian_3x3": {
        "family": "V1",
        "preprocessing": "gaussian_3x3",
        "morphology": "none",
    },
    "V1_gaussian_5x5": {
        "family": "V1",
        "preprocessing": "gaussian_5x5",
        "morphology": "none",
    },
    "V2_median_3x3": {
        "family": "V2",
        "preprocessing": "median_3x3",
        "morphology": "none",
    },
    "V2_median_5x5": {
        "family": "V2",
        "preprocessing": "median_5x5",
        "morphology": "none",
    },
    "V3_opening_3x3": {
        "family": "V3",
        "preprocessing": "none",
        "morphology": "opening_3x3",
    },
    "V3_closing_3x3": {
        "family": "V3",
        "preprocessing": "none",
        "morphology": "closing_3x3",
    },
    "V3_opening_closing_3x3": {
        "family": "V3",
        "preprocessing": "none",
        "morphology": "opening_closing_3x3",
    },
}

SMOOTHING_VARIANTS = {
    "V5_score_median3": {"family": "V5", "smoothing": "median3"},
    "V5_score_mean3": {"family": "V5", "smoothing": "mean3"},
}

BASELINE_REFERENCE = {
    "roc_auc": 0.8817395852793477,
    "pr_auc": 0.8311554406587465,
    "threshold_directed": 0.08747855917667238,
    "recall": 0.7291252485089463,
    "fpr": 0.060542309490416085,
}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer CSV header for empty rows: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "max_ms": float(np.max(array)),
    }


def safe_absdiff(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Subtracao segura para uint16: nunca faz underflow antes do absdiff."""

    return np.abs(
        current.astype(np.float32, copy=False) - previous.astype(np.float32, copy=False)
    )


def preprocess(frame: np.ndarray, kind: str) -> np.ndarray:
    """Pre-processamento espacial fixo; retorna shape igual e dtype float32."""

    source = frame.astype(np.float32, copy=False)
    if kind == "none":
        return source
    if kind == "gaussian_3x3":
        return ndimage.gaussian_filter(source, sigma=0.8, radius=1, mode="nearest")
    if kind == "gaussian_5x5":
        return ndimage.gaussian_filter(source, sigma=1.1, radius=2, mode="nearest")
    if kind == "median_3x3":
        return ndimage.median_filter(source, size=3, mode="nearest")
    if kind == "median_5x5":
        return ndimage.median_filter(source, size=5, mode="nearest")
    raise ValueError(f"unknown pre-processing: {kind}")


def morph_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    """Morfologia binaria minima, aplicada somente apos threshold de diff."""

    if kind == "none":
        return mask.astype(bool, copy=False)
    structure = np.ones((3, 3), dtype=bool)
    if kind == "opening_3x3":
        return ndimage.binary_opening(mask, structure=structure)
    if kind == "closing_3x3":
        return ndimage.binary_closing(mask, structure=structure)
    if kind == "opening_closing_3x3":
        opened = ndimage.binary_opening(mask, structure=structure)
        return ndimage.binary_closing(opened, structure=structure)
    raise ValueError(f"unknown morphology: {kind}")


def pdi_score_processed(
    previous_processed: np.ndarray, current_processed: np.ndarray, morphology: str
) -> float:
    """Score baseline sobre frames já pré-processados, na ROI B."""

    region = base.roi_slices(current_processed.shape, ROI_B)
    mask = safe_absdiff(previous_processed[region], current_processed[region]) >= PIXEL_THRESHOLD_MM
    mask = morph_mask(mask, morphology)
    return float(prior.component_geometry(mask)["largest_component_changed_fraction"])


def pdi_score(previous: np.ndarray, current: np.ndarray, specification: dict) -> float:
    """Convenience wrapper usada pelos testes e pelo microbenchmark."""

    return pdi_score_processed(
        preprocess(previous, specification["preprocessing"]),
        preprocess(current, specification["preprocessing"]),
        specification["morphology"],
    )


def smooth_score(history: deque[float], score: float, kind: str) -> float:
    """Suavizacao causal: somente scores ate o evento atual sao observados."""

    history.append(float(score))
    values = np.asarray(history, dtype=np.float64)
    if kind == "median3":
        return float(np.median(values))
    if kind == "mean3":
        return float(np.mean(values))
    raise ValueError(f"unknown score smoothing: {kind}")


def score_pairs(
    data_root: Path,
    indexes: dict[str, list[dict]],
    variants: dict[str, dict],
) -> dict[str, dict[int, dict[str, float]]]:
    """Calcula scores por par fisico adjacente uma unica vez por variante."""

    output: dict[str, dict[int, dict[str, float]]] = {}
    for passage_number, (passage_id, rows) in enumerate(indexes.items(), start=1):
        frames = [
            base.read_depth(data_root / "DEPTH" / passage_id / row["depth_filename"])
            for row in rows
        ]
        # Cada frame aparece em até dois pares consecutivos. Cache local evita
        # recomputar o mesmo blur/median e não muda o resultado do PDI.
        preprocessed = {
            kind: [preprocess(frame, kind) for frame in frames]
            for kind in sorted({specification["preprocessing"] for specification in variants.values()})
        }
        per_capture: dict[int, dict[str, float]] = {}
        for capture_index in range(2, len(frames) + 1):
            per_capture[capture_index] = {
                variant: pdi_score_processed(
                    preprocessed[specification["preprocessing"]][capture_index - 2],
                    preprocessed[specification["preprocessing"]][capture_index - 1],
                    specification["morphology"],
                )
                for variant, specification in variants.items()
            }
        output[passage_id] = per_capture
        if passage_number % 25 == 0:
            print(f"scores: {passage_number}/{len(indexes)} passages", flush=True)
    return output


def build_series(
    indexes: dict[str, list[dict]],
    p99_lookup: dict[tuple[str, int], float],
    pair_scores: dict[str, dict[int, dict[str, float]]],
    static_variants: Iterable[str],
    smoothing_variants: dict[str, dict] | None = None,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Aplica gate e reset por passagem; baseline apos INVALID recebe NaN."""

    output: dict[str, dict[str, dict[str, list[float]]]] = {}
    all_variants = list(static_variants) + list((smoothing_variants or {}).keys())
    for gate_mode in ("predicted_p99", "oracle_label"):
        output[gate_mode] = {variant: {} for variant in all_variants}
        for passage_id, rows in indexes.items():
            histories = {
                variant: deque(maxlen=3) for variant in (smoothing_variants or {})
            }
            has_previous_valid = False
            passage_values = {variant: [] for variant in all_variants}
            for capture_index, row in enumerate(rows, start=1):
                invalid = (
                    p99_lookup[(passage_id, capture_index)] >= QUALITY_P99_THRESHOLD_MM
                    if gate_mode == "predicted_p99"
                    else row["label"] == "ruido"
                )
                if invalid:
                    has_previous_valid = False
                    for history in histories.values():
                        history.clear()
                    for values in passage_values.values():
                        values.append(float("nan"))
                    continue
                if not has_previous_valid:
                    has_previous_valid = True
                    for history in histories.values():
                        history.clear()
                    for values in passage_values.values():
                        values.append(float("nan"))
                    continue
                raw = pair_scores[passage_id][capture_index]
                for variant in static_variants:
                    passage_values[variant].append(raw[variant])
                for variant, specification in (smoothing_variants or {}).items():
                    passage_values[variant].append(
                        smooth_score(histories[variant], raw["V0_baseline"], specification["smoothing"])
                    )
            for variant in all_variants:
                output[gate_mode][variant][passage_id] = passage_values[variant]
    return output


def rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate((positive, negative))
    ranks = stats.rankdata(values)
    n_positive, n_negative = len(positive), len(negative)
    return float(
        (ranks[:n_positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    positives = int(np.sum(ordered))
    return float(np.sum(precision[ordered]) / positives) if positives else 0.0


def operating_point(labels: np.ndarray, scores: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(~sorted_labels)
    tpr = tp / int(np.sum(labels))
    fpr = fp / int(np.sum(~labels))
    distinct = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    candidates = np.flatnonzero(distinct)
    selected = candidates[np.argmax((tpr - fpr)[candidates])]
    return {
        "threshold_directed": float(sorted_scores[selected]),
        "recall": float(tpr[selected]),
        "fpr": float(fpr[selected]),
    }


def valid_target_arrays(
    indexes: dict[str, list[dict]], series: dict[str, list[float]]
) -> tuple[np.ndarray, np.ndarray]:
    labels, values = [], []
    for passage_id, frames in indexes.items():
        for frame, score in zip(frames, series[passage_id]):
            if frame["label"] not in {"background", "parcial", "suited"} or not math.isfinite(score):
                continue
            labels.append(frame["label"] in {"parcial", "suited"})
            values.append(score)
    return np.asarray(labels, dtype=bool), np.asarray(values, dtype=np.float64)


def frame_metrics(
    indexes: dict[str, list[dict]],
    series: dict[str, list[float]],
    direction: float | None = None,
    threshold_directed: float | None = None,
) -> dict:
    labels, raw_values = valid_target_arrays(indexes, series)
    raw_auc = rank_auc(raw_values[labels], raw_values[~labels])
    direction = direction if direction is not None else (1.0 if raw_auc >= 0.5 else -1.0)
    scores = raw_values * direction
    point = operating_point(labels, scores)
    threshold = point["threshold_directed"] if threshold_directed is None else threshold_directed
    predicted = scores >= threshold
    return {
        "n_positive": int(np.sum(labels)),
        "n_negative": int(np.sum(~labels)),
        "direction": direction,
        "roc_auc": raw_auc if direction > 0 else 1.0 - raw_auc,
        "pr_auc": average_precision(labels, scores),
        "threshold_directed": threshold,
        "recall": float(np.mean(predicted[labels])),
        "fpr": float(np.mean(predicted[~labels])),
    }


def operational_metrics(
    indexes: dict[str, list[dict]],
    series: dict[str, list[float]],
    direction: float,
    threshold: float,
) -> dict:
    per_passage = []
    for passage_id, frames in indexes.items():
        state, no_motion = False, 0
        pre_states, post_states, activation_indices = [], [], []
        for index, score in enumerate(series[passage_id]):
            pre_states.append(state)
            if math.isfinite(score):
                moving = score * direction >= threshold
                previous_state = state
                if moving:
                    state, no_motion = True, 0
                elif state:
                    no_motion += 1
                    if no_motion >= IDLE_PATIENCE:
                        state, no_motion = False, 0
                if state and not previous_state:
                    activation_indices.append(index)
            post_states.append(state)

        labels = [frame["label"] for frame in frames]
        timestamps = np.asarray([float(frame["relative_time_ms"]) for frame in frames])
        suited = [index for index, label in enumerate(labels) if label == "suited"]
        background = [index for index, label in enumerate(labels) if label == "background"]
        intervals = np.diff(timestamps)
        first_suited = suited[0] if suited else None
        first_activation = activation_indices[0] if activation_indices else None
        per_passage.append(
            {
                "n_suited": len(suited),
                "suited_forward_active": sum(pre_states[index] for index in suited),
                "suited_passage_covered": bool(any(pre_states[index] for index in suited)),
                "activated_before_suited": bool(
                    first_activation is not None and first_suited is not None and first_activation < first_suited
                ),
                "activation_in_partial": bool(
                    first_activation is not None and labels[first_activation] == "parcial"
                ),
                "activation_delay_ms": (
                    None
                    if first_activation is None or first_suited is None
                    else float(timestamps[first_activation] - timestamps[first_suited])
                ),
                "false_active_background": sum(post_states[index] for index in background),
                "background_frames": len(background),
                "active_time_ms": float(np.sum(intervals[np.asarray(post_states[:-1], dtype=bool)])),
                "total_time_ms": float(np.sum(intervals)),
            }
        )

    suited_passages = [row for row in per_passage if row["n_suited"]]
    delays = [row["activation_delay_ms"] for row in suited_passages if row["activation_delay_ms"] is not None]
    total_suited = sum(row["n_suited"] for row in suited_passages)
    total_background = sum(row["background_frames"] for row in per_passage)
    total_time = sum(row["total_time_ms"] for row in per_passage)
    return {
        "operational_coverage": sum(row["suited_passage_covered"] for row in suited_passages) / len(suited_passages),
        "suited_retention": sum(row["suited_forward_active"] for row in suited_passages) / total_suited,
        "activation_before_suited": sum(row["activated_before_suited"] for row in suited_passages) / len(suited_passages),
        "activation_in_partial": sum(row["activation_in_partial"] for row in suited_passages) / len(suited_passages),
        "activation_delay_median_ms": float(np.median(delays)) if delays else None,
        "false_active_background": sum(row["false_active_background"] for row in per_passage) / total_background,
        "time_active_ratio": sum(row["active_time_ms"] for row in per_passage) / total_time,
    }


def evaluate_variants(
    indexes: dict[str, list[dict]],
    series: dict[str, dict[str, dict[str, list[float]]]],
    variants: dict[str, dict],
) -> list[dict]:
    output = []
    for variant, configuration in variants.items():
        # Regra herdada do baseline: escolha do threshold pelo Youden usando
        # quality gate oraculo; relato operacional com gate P99 executavel.
        oracle = frame_metrics(indexes, series["oracle_label"][variant])
        predicted = frame_metrics(
            indexes,
            series["predicted_p99"][variant],
            direction=oracle["direction"],
            threshold_directed=oracle["threshold_directed"],
        )
        operational = operational_metrics(
            indexes,
            series["predicted_p99"][variant],
            oracle["direction"],
            oracle["threshold_directed"],
        )
        output.append(
            {
                "variant": variant,
                **configuration,
                "threshold_selection": "oracle_label Youden (baseline rule)",
                "threshold_directed": oracle["threshold_directed"],
                "direction": oracle["direction"],
                "roc_auc": predicted["roc_auc"],
                "pr_auc": predicted["pr_auc"],
                "recall": predicted["recall"],
                "fpr": predicted["fpr"],
                **operational,
            }
        )
    return output


def choose_best_morphology(results: list[dict]) -> str:
    candidates = [row for row in results if row["family"] == "V3"]
    winner = max(
        candidates,
        key=lambda row: (row["roc_auc"], row["pr_auc"], -row["fpr"]),
    )
    return winner["morphology"]


def collect_benchmark_pairs(
    data_root: Path, indexes: dict[str, list[dict]], limit: int = BENCHMARK_PAIRS):
    pairs = []
    for passage_id, rows in indexes.items():
        previous = None
        for row in rows:
            if row["label"] == "ruido":
                previous = None
                continue
            current = base.read_depth(data_root / "DEPTH" / passage_id / row["depth_filename"])
            if previous is not None:
                pairs.append((previous, current))
                if len(pairs) == limit:
                    return pairs
            previous = current
    return pairs


def benchmark_variant(pairs, specification: dict, smoothing: str | None = None) -> dict:
    phases = defaultdict(list)
    history: deque[float] = deque(maxlen=3)
    for previous, current in pairs:
        started = time.perf_counter_ns()
        previous_processed = preprocess(previous, specification["preprocessing"])
        current_processed = preprocess(current, specification["preprocessing"])
        pre_end = time.perf_counter_ns()
        region = base.roi_slices(current.shape, ROI_B)
        mask = safe_absdiff(previous_processed[region], current_processed[region]) >= PIXEL_THRESHOLD_MM
        mask = morph_mask(mask, specification["morphology"])
        mask_end = time.perf_counter_ns()
        score = float(prior.component_geometry(mask)["largest_component_changed_fraction"])
        component_end = time.perf_counter_ns()
        if smoothing:
            _ = smooth_score(history, score, smoothing)
        smoothing_end = time.perf_counter_ns()
        phases["preprocessing"].append((pre_end - started) / 1_000_000.0)
        phases["absdiff_mask"].append((mask_end - pre_end) / 1_000_000.0)
        phases["components"].append((component_end - mask_end) / 1_000_000.0)
        phases["smoothing"].append((smoothing_end - component_end) / 1_000_000.0)
        phases["total"].append((smoothing_end - started) / 1_000_000.0)
    summary = {phase: quantiles(values) for phase, values in phases.items()}
    return {
        "mean_latency_ms": summary["total"]["mean_ms"],
        "median_latency_ms": summary["total"]["median_ms"],
        "p95_latency_ms": summary["total"]["p95_ms"],
        "p99_latency_ms": summary["total"]["p99_ms"],
        "ops_per_s": 1000.0 / summary["total"]["mean_ms"],
        "phase_summary": summary,
    }


def benchmark_variants(data_root: Path, indexes: dict[str, list[dict]], variants: dict[str, dict]) -> list[dict]:
    pairs = collect_benchmark_pairs(data_root, indexes)
    output = []
    for variant, configuration in variants.items():
        timing = benchmark_variant(pairs, configuration, configuration.get("smoothing"))
        output.append(
            {
                "variant": variant,
                **configuration,
                "samples": len(pairs),
                **{key: value for key, value in timing.items() if key != "phase_summary"},
                "preprocessing_mean_ms": timing["phase_summary"]["preprocessing"]["mean_ms"],
                "absdiff_mask_mean_ms": timing["phase_summary"]["absdiff_mask"]["mean_ms"],
                "components_mean_ms": timing["phase_summary"]["components"]["mean_ms"],
                "smoothing_mean_ms": timing["phase_summary"]["smoothing"]["mean_ms"],
            }
        )
    return output


def assert_baseline(results: list[dict]) -> dict:
    baseline = next(row for row in results if row["variant"] == "V0_baseline")
    mismatches = {
        key: {"observed": baseline[key], "reference": expected}
        for key, expected in BASELINE_REFERENCE.items()
        if not math.isclose(float(baseline[key]), expected, rel_tol=0.0, abs_tol=1e-10)
    }
    if mismatches:
        raise AssertionError(f"baseline reproduction mismatch: {mismatches}")
    return baseline


def merge_latency(results: list[dict], latency: list[dict]) -> list[dict]:
    lookup = {row["variant"]: row for row in latency}
    baseline = lookup["V0_baseline"]["mean_latency_ms"]
    output = []
    for row in results:
        timing = lookup[row["variant"]]
        output.append(
            {
                **row,
                "mean_latency_ms": timing["mean_latency_ms"],
                "median_latency_ms": timing["median_latency_ms"],
                "p95_latency_ms": timing["p95_latency_ms"],
                "p99_latency_ms": timing["p99_latency_ms"],
                "ops_per_s": timing["ops_per_s"],
                "relative_cost": timing["mean_latency_ms"] / baseline,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--cohort-metrics", type=Path, default=base.DEFAULT_COHORT_METRICS)
    parser.add_argument("--frame-features", type=Path, default=DEFAULT_FRAME_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    passage_ids = base.load_cohort(args.cohort_metrics)
    indexes = base.load_indexes(args.data_root, passage_ids)
    observed = (
        len(indexes),
        sum(len(rows) for rows in indexes.values()),
        sum(row["label"] == "suited" for rows in indexes.values() for row in rows),
    )
    if observed != EXPECTED_COHORT:
        raise ValueError(f"operational cohort mismatch: {observed} != {EXPECTED_COHORT}")
    p99_lookup = classical.load_frame_p99(args.frame_features)

    started = time.perf_counter()
    pair_scores = score_pairs(args.data_root, indexes, PHASE_ONE_VARIANTS)
    phase_one_series = build_series(
        indexes,
        p99_lookup,
        pair_scores,
        PHASE_ONE_VARIANTS,
        smoothing_variants=SMOOTHING_VARIANTS,
    )
    phase_one_config = {**PHASE_ONE_VARIANTS, **SMOOTHING_VARIANTS}
    phase_one_results = evaluate_variants(indexes, phase_one_series, phase_one_config)
    baseline = assert_baseline(phase_one_results)

    best_morphology = choose_best_morphology(phase_one_results)
    v4_name = f"V4_gaussian_3x3_plus_{best_morphology}"
    v4_config = {
        v4_name: {
            "family": "V4",
            "preprocessing": "gaussian_3x3",
            "morphology": best_morphology,
        }
    }
    v4_pair_scores = score_pairs(args.data_root, indexes, v4_config)
    v4_series = build_series(indexes, p99_lookup, v4_pair_scores, v4_config)
    v4_results = evaluate_variants(indexes, v4_series, v4_config)
    results = phase_one_results + v4_results

    full_config = {**phase_one_config, **v4_config}
    # Smoothing usa o score base; para benchmark, a mesma geracao base e medida.
    benchmark_config = {
        name: (
            {"preprocessing": "none", "morphology": "none", **config}
            if config["family"] == "V5"
            else config
        )
        for name, config in full_config.items()
    }
    latency = benchmark_variants(args.data_root, indexes, benchmark_config)
    results = merge_latency(results, latency)
    results.sort(key=lambda row: (-row["roc_auc"], -row["pr_auc"], row["mean_latency_ms"]))

    args.output.mkdir(parents=True, exist_ok=True)
    result_fields = [
        "variant", "family", "preprocessing", "morphology", "smoothing",
        "threshold_selection", "threshold_directed", "direction", "roc_auc", "pr_auc",
        "recall", "fpr", "operational_coverage", "suited_retention",
        "activation_before_suited", "activation_in_partial", "activation_delay_median_ms",
        "false_active_background", "time_active_ratio", "mean_latency_ms",
        "median_latency_ms", "p95_latency_ms", "p99_latency_ms", "ops_per_s", "relative_cost",
    ]
    for row in results:
        row.setdefault("preprocessing", "none")
        row.setdefault("morphology", "none")
        row.setdefault("smoothing", "none")
    write_csv(args.output / "variant_results.csv", results, result_fields)
    write_csv(args.output / "microbenchmark.csv", latency)

    summary = {
        "scope": "offline PDI preprocessing ablation; no runtime/model changes",
        "cohort": {"passages": observed[0], "frames": observed[1], "suited_frames": observed[2]},
        "target": "parcial+suited positive; background negative; ruido reset by quality gate",
        "quality_gate": {
            "feature": "depth_p99_mm",
            "threshold_mm": QUALITY_P99_THRESHOLD_MM,
            "semantics": "INVALID clears previous; next VALID is baseline; no temporal comparison crosses invalid",
        },
        "baseline_reproduction": baseline,
        "variants_tested": len(results),
        "phase_one_variants": list(PHASE_ONE_VARIANTS),
        "smoothing_variants": list(SMOOTHING_VARIANTS),
        "variant_specifications": full_config,
        "filter_parameters": {
            "gaussian_3x3": {"sigma": 0.8, "radius": 1, "mode": "nearest"},
            "gaussian_5x5": {"sigma": 1.1, "radius": 2, "mode": "nearest"},
            "median_3x3": {"size": 3, "mode": "nearest"},
            "median_5x5": {"size": 5, "mode": "nearest"},
            "morphology": {"structure": "3x3 all-ones binary"},
            "score_smoothing": {"window": 3, "causal": True},
        },
        "selected_v3_morphology_for_v4": best_morphology,
        "downsampling": "not tested; outside the necessary preprocessing-only comparison",
        "benchmark": "local Mac ranking only; requires Raspberry Pi confirmation",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "configuration.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
