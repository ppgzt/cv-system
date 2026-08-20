#!/usr/bin/env python3
"""Comparacao offline de sinais classicos de PDI para o Visual Event.

Este script e deliberadamente separado do runtime. Ele nao importa agentes PADE,
nao altera o detector online e nao treina modelos. O quality gate usa o sinal
previamente auditado (P99 depth) e sempre quebra a continuidade temporal:

    INVALID -> limpa historico
    proximo VALID -> baseline, sem score temporal
    VALID seguinte -> volta a produzir score

Uma segunda leitura usa o label humano ``ruido`` como gate-oraculo para separar
o limite do sinal de movimento do erro residual do quality gate executavel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, stats
from skimage.registration import optical_flow_ilk


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import visual_event_diagnostic as base  # noqa: E402
import visual_event_noise_pdi_audit as prior  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "data-analysis/visual_event_classical_pdi_output"
DEFAULT_FRAME_FEATURES = (
    REPO_ROOT / "data-analysis/visual_event_noise_pdi_output/frame_features.csv"
)
EXPECTED_COHORT = (184, 13_741, 1_655)
ROI_B = (0.30, 0.70, 0.20, 0.80)
PIXEL_THRESHOLD_MM = 200.0
QUALITY_P99_THRESHOLD_MM = 2230.0
IDLE_PATIENCE = 3
FLOW_DOWNSAMPLE = 2
FLOW_RADIUS = 3
FLOW_NUM_WARP = 1
FLOW_MAGNITUDE_MIN_PX = 1.0
MHI_WINDOWS = (2, 3, 5)
ADAPTIVE_WINDOW = 3


METHOD_FEATURES = {
    "baseline_components": ("baseline_component_coherence",),
    "optical_flow_ilk": (
        "flow_mean_magnitude_px",
        "flow_median_magnitude_px",
        "flow_p90_magnitude_px",
        "flow_mean_abs_horizontal_px",
        "flow_p90_abs_horizontal_px",
        "flow_fraction_above_1px",
        "flow_directional_coherence",
        "flow_horizontal_energy",
        "flow_horizontal_sign_coherence",
        "flow_dominant_direction_fraction",
        "flow_coherent_horizontal_energy",
    ),
    "three_frame_differencing": (
        "three_intersection_changed_ratio",
        "three_intersection_largest_area_ratio",
        "three_intersection_component_coherence",
        "three_union_changed_ratio",
        "three_union_largest_area_ratio",
    ),
    "motion_history": tuple(
        f"mhi_w{window}_{suffix}"
        for window in MHI_WINDOWS
        for suffix in (
            "active_ratio",
            "persistent_ratio",
            "weighted_mean",
            "largest_area_ratio",
            "component_coherence",
        )
    ),
    "adaptive_background": (
        "adaptive_changed_ratio",
        "adaptive_largest_area_ratio",
        "adaptive_component_coherence",
        "adaptive_diff_median_mm",
        "adaptive_diff_p90_mm",
    ),
}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer CSV header for empty rows: {path}")
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def ns_to_ms(value: int) -> float:
    return value / 1_000_000.0


def quantiles(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def load_frame_p99(path: Path) -> dict[tuple[str, int], float]:
    return {
        (row["passage_id"], int(row["capture_index"])): float(row["depth_p99_mm"])
        for row in read_csv(path)
    }


def downsample_mean(frame: np.ndarray, factor: int = FLOW_DOWNSAMPLE) -> np.ndarray:
    height = frame.shape[0] // factor * factor
    width = frame.shape[1] // factor * factor
    cropped = frame[:height, :width]
    return cropped.reshape(height // factor, factor, width // factor, factor).mean(
        axis=(1, 3), dtype=np.float32
    )


def component_features(mask: np.ndarray) -> dict:
    geometry = prior.component_geometry(mask)
    return {
        "changed_ratio": float(np.mean(mask)),
        "largest_area_ratio": float(geometry["largest_component_area_ratio"]),
        "component_coherence": float(
            geometry["largest_component_changed_fraction"]
        ),
    }


def baseline_features(previous: np.ndarray, current: np.ndarray) -> dict:
    region = base.roi_slices(current.shape, ROI_B)
    diff = np.abs(
        current[region].astype(np.int32, copy=False)
        - previous[region].astype(np.int32, copy=False)
    )
    features = component_features(diff >= PIXEL_THRESHOLD_MM)
    return {
        "baseline_changed_ratio": features["changed_ratio"],
        "baseline_largest_area_ratio": features["largest_area_ratio"],
        "baseline_component_coherence": features["component_coherence"],
    }


def flow_features(previous: np.ndarray, current: np.ndarray) -> tuple[dict, np.ndarray]:
    region = base.roi_slices(current.shape, ROI_B)
    previous_small = downsample_mean(previous[region].astype(np.float32, copy=False))
    current_small = downsample_mean(current[region].astype(np.float32, copy=False))
    # Escala fixa compartilhada: preserva constancia relativa sem normalizar cada
    # frame independentemente, o que criaria movimento artificial.
    previous_small /= 2600.0
    current_small /= 2600.0
    flow = optical_flow_ilk(
        previous_small,
        current_small,
        radius=FLOW_RADIUS,
        num_warp=FLOW_NUM_WARP,
        gaussian=True,
        prefilter=True,
        dtype=np.float32,
    )
    vertical, horizontal = flow
    magnitude = np.hypot(horizontal, vertical)
    moving = magnitude >= FLOW_MAGNITUDE_MIN_PX
    magnitude_sum = float(np.sum(magnitude, dtype=np.float64))
    vector_resultant = float(
        math.hypot(
            float(np.sum(horizontal, dtype=np.float64)),
            float(np.sum(vertical, dtype=np.float64)),
        )
    )
    directional_coherence = vector_resultant / magnitude_sum if magnitude_sum else 0.0
    horizontal_abs_sum = float(np.sum(np.abs(horizontal), dtype=np.float64))
    horizontal_sign_coherence = (
        abs(float(np.sum(horizontal, dtype=np.float64))) / horizontal_abs_sum
        if horizontal_abs_sum
        else 0.0
    )
    horizontal_fraction = float(
        np.mean(np.abs(horizontal) / np.maximum(magnitude, 1e-6))
    )
    if np.any(moving):
        dominant_angle = math.atan2(
            float(np.sum(vertical[moving], dtype=np.float64)),
            float(np.sum(horizontal[moving], dtype=np.float64)),
        )
        angular_distance = np.cos(np.arctan2(vertical[moving], horizontal[moving]) - dominant_angle)
        dominant_fraction = float(np.mean(angular_distance >= math.cos(math.radians(30))))
    else:
        dominant_fraction = 0.0
    p90 = float(np.quantile(magnitude, 0.90))
    mean_horizontal = float(np.mean(horizontal))
    mean_vertical = float(np.mean(vertical))
    dominant_angle_degrees = float(
        math.degrees(math.atan2(mean_vertical, mean_horizontal))
    )
    if abs(mean_horizontal) >= abs(mean_vertical):
        dominant_orientation = "right" if mean_horizontal >= 0 else "left"
    else:
        dominant_orientation = "down" if mean_vertical >= 0 else "up"
    return (
        {
            "flow_mean_magnitude_px": float(np.mean(magnitude)),
            "flow_median_magnitude_px": float(np.median(magnitude)),
            "flow_p90_magnitude_px": p90,
            "flow_signed_mean_horizontal_px": mean_horizontal,
            "flow_mean_abs_horizontal_px": float(np.mean(np.abs(horizontal))),
            "flow_p90_abs_horizontal_px": float(np.quantile(np.abs(horizontal), 0.90)),
            "flow_signed_mean_vertical_px": mean_vertical,
            "flow_dominant_angle_degrees": dominant_angle_degrees,
            "flow_dominant_orientation": dominant_orientation,
            "flow_fraction_above_1px": float(np.mean(moving)),
            "flow_directional_coherence": directional_coherence,
            "flow_horizontal_energy": p90 * horizontal_fraction,
            "flow_horizontal_sign_coherence": horizontal_sign_coherence,
            "flow_dominant_direction_fraction": dominant_fraction,
            "flow_coherent_horizontal_energy": (
                p90 * horizontal_fraction * directional_coherence
            ),
        },
        flow,
    )


def three_frame_features(
    previous_previous: np.ndarray, previous: np.ndarray, current: np.ndarray
) -> dict:
    region = base.roi_slices(current.shape, ROI_B)
    older = previous_previous[region].astype(np.int32, copy=False)
    middle = previous[region].astype(np.int32, copy=False)
    newer = current[region].astype(np.int32, copy=False)
    first = np.abs(middle - older) >= PIXEL_THRESHOLD_MM
    second = np.abs(newer - middle) >= PIXEL_THRESHOLD_MM
    intersection = component_features(first & second)
    union = component_features(first | second)
    return {
        "three_intersection_changed_ratio": intersection["changed_ratio"],
        "three_intersection_largest_area_ratio": intersection["largest_area_ratio"],
        "three_intersection_component_coherence": intersection["component_coherence"],
        "three_union_changed_ratio": union["changed_ratio"],
        "three_union_largest_area_ratio": union["largest_area_ratio"],
    }


def motion_history_features(mask_history: deque[np.ndarray]) -> dict:
    output = {}
    masks = list(mask_history)
    for window in MHI_WINDOWS:
        if len(masks) < window:
            for suffix in (
                "active_ratio",
                "persistent_ratio",
                "weighted_mean",
                "largest_area_ratio",
                "component_coherence",
            ):
                output[f"mhi_w{window}_{suffix}"] = float("nan")
            continue
        selected = np.asarray(masks[-window:], dtype=np.float32)
        counts = np.sum(selected, axis=0)
        active = counts > 0
        persistent = counts >= 2
        weights = np.arange(1, window + 1, dtype=np.float32)
        weighted = np.tensordot(weights, selected, axes=(0, 0)) / float(weights.sum())
        geometry = component_features(active)
        output.update(
            {
                f"mhi_w{window}_active_ratio": float(np.mean(active)),
                f"mhi_w{window}_persistent_ratio": float(np.mean(persistent)),
                f"mhi_w{window}_weighted_mean": float(np.mean(weighted)),
                f"mhi_w{window}_largest_area_ratio": geometry["largest_area_ratio"],
                f"mhi_w{window}_component_coherence": geometry["component_coherence"],
            }
        )
    return output


def adaptive_background_features(history: deque[np.ndarray], current: np.ndarray) -> dict:
    if len(history) < ADAPTIVE_WINDOW:
        return {feature: float("nan") for feature in METHOD_FEATURES["adaptive_background"]}
    region = base.roi_slices(current.shape, ROI_B)
    baseline = np.median(
        np.stack([frame[region] for frame in list(history)[-ADAPTIVE_WINDOW:]]),
        axis=0,
    )
    diff = np.abs(current[region].astype(np.float32, copy=False) - baseline)
    geometry = component_features(diff >= PIXEL_THRESHOLD_MM)
    return {
        "adaptive_changed_ratio": geometry["changed_ratio"],
        "adaptive_largest_area_ratio": geometry["largest_area_ratio"],
        "adaptive_component_coherence": geometry["component_coherence"],
        "adaptive_diff_median_mm": float(np.median(diff)),
        "adaptive_diff_p90_mm": float(np.quantile(diff, 0.90)),
    }


def stable_background_flags(rows: list[dict]) -> list[bool]:
    labels = [row["label"] for row in rows]
    flags = []
    for index, label in enumerate(labels):
        low = max(0, index - 2)
        high = min(len(labels), index + 3)
        flags.append(
            label == "background"
            and high - low == 5
            and all(item == "background" for item in labels[low:high])
        )
    return flags


def timed_call(samples: dict[str, list[dict]], method: str, function, *args):
    start = time.perf_counter_ns()
    result = function(*args)
    end = time.perf_counter_ns()
    samples[method].append(
        {
            "preprocessing_ms": 0.0,
            "algorithm_ms": ns_to_ms(end - start),
            "decision_ms": 0.0,
            "total_ms": ns_to_ms(end - start),
        }
    )
    return result


def extract_features(
    data_root: Path,
    indexes: dict[str, list[dict]],
    p99_lookup: dict[tuple[str, int], float],
) -> tuple[list[dict], dict[str, list[dict]], dict]:
    output: list[dict] = []
    timings: dict[str, list[dict]] = defaultdict(list)
    quality_counts = defaultdict(int)

    for passage_number, (passage_id, rows) in enumerate(indexes.items(), start=1):
        frames = [
            base.read_depth(data_root / "DEPTH" / passage_id / row["depth_filename"])
            for row in rows
        ]
        stable_flags = stable_background_flags(rows)
        pair_cache = {}
        for gate_mode in ("predicted_p99", "oracle_label"):
            history: deque[np.ndarray] = deque(maxlen=max(MHI_WINDOWS) + 1)
            mask_history: deque[np.ndarray] = deque(maxlen=max(MHI_WINDOWS))
            for index, (row, current) in enumerate(zip(rows, frames), start=1):
                p99 = p99_lookup[(passage_id, index)]
                predicted_invalid = p99 >= QUALITY_P99_THRESHOLD_MM
                oracle_invalid = row["label"] == "ruido"
                invalid = predicted_invalid if gate_mode == "predicted_p99" else oracle_invalid
                quality_counts[(gate_mode, "invalid" if invalid else "valid")] += 1
                metadata = {
                    "gate_mode": gate_mode,
                    "passage_id": passage_id,
                    "capture_index": index,
                    "relative_time_ms": float(row["relative_time_ms"]),
                    "label": row["label"],
                    "depth_filename": row["depth_filename"],
                    "depth_p99_mm": p99,
                    "predicted_invalid": predicted_invalid,
                    "oracle_invalid": oracle_invalid,
                    "stable_background": stable_flags[index - 1],
                    "has_temporal_score": False,
                }
                if invalid:
                    history.clear()
                    mask_history.clear()
                    output.append(metadata)
                    continue

                if history:
                    previous = history[-1]
                    cache_key = index
                    if cache_key not in pair_cache:
                        baseline = timed_call(
                            timings, "baseline_components", baseline_features, previous, current
                        )
                        flow, _ = timed_call(
                            timings, "optical_flow_ilk", flow_features, previous, current
                        )
                        pair_cache[cache_key] = {**baseline, **flow}
                    metadata.update(pair_cache[cache_key])
                    metadata["has_temporal_score"] = True

                    region = base.roi_slices(current.shape, ROI_B)
                    current_mask = (
                        np.abs(
                            current[region].astype(np.int32, copy=False)
                            - previous[region].astype(np.int32, copy=False)
                        )
                        >= PIXEL_THRESHOLD_MM
                    )
                    mask_history.append(current_mask)
                    mhi_start = time.perf_counter_ns()
                    metadata.update(motion_history_features(mask_history))
                    mhi_end = time.perf_counter_ns()
                    timings["motion_history"].append(
                        {
                            "preprocessing_ms": 0.0,
                            "algorithm_ms": ns_to_ms(mhi_end - mhi_start),
                            "decision_ms": 0.0,
                            "total_ms": ns_to_ms(mhi_end - mhi_start),
                        }
                    )
                    if len(history) >= 2:
                        metadata.update(
                            timed_call(
                                timings,
                                "three_frame_differencing",
                                three_frame_features,
                                history[-2],
                                previous,
                                current,
                            )
                        )
                    else:
                        for feature in METHOD_FEATURES["three_frame_differencing"]:
                            metadata[feature] = float("nan")

                    metadata.update(
                        timed_call(
                            timings,
                            "adaptive_background",
                            adaptive_background_features,
                            history,
                            current,
                        )
                    )
                output.append(metadata)
                history.append(current)

        if passage_number % 25 == 0:
            print(f"features: {passage_number}/{len(indexes)} passages", flush=True)

    quality_summary = {
        f"{gate_mode}_{status}": count
        for (gate_mode, status), count in sorted(quality_counts.items())
    }
    return output, timings, quality_summary


def rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate((positive, negative))
    ranks = stats.rankdata(values)
    n_positive = len(positive)
    n_negative = len(negative)
    return float(
        (ranks[:n_positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    true_positive = np.cumsum(ordered)
    precision = true_positive / np.arange(1, len(labels) + 1)
    positives = int(np.sum(ordered))
    return float(np.sum(precision[ordered]) / positives) if positives else 0.0


def operating_point(labels: np.ndarray, scores: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(~sorted_labels)
    positives = int(np.sum(labels))
    negatives = len(labels) - positives
    tpr = tp / positives
    fpr = fp / negatives
    distinct = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    candidates = np.flatnonzero(distinct)
    selected = candidates[np.argmax((tpr - fpr)[candidates])]
    return {
        "threshold_directed": float(sorted_scores[selected]),
        "recall": float(tpr[selected]),
        "fpr": float(fpr[selected]),
    }


def feature_performance(rows: list[dict]) -> list[dict]:
    output = []
    for gate_mode in ("predicted_p99", "oracle_label"):
        gate_rows = [
            row
            for row in rows
            if row["gate_mode"] == gate_mode
            and row["label"] in {"background", "parcial", "suited"}
        ]
        for background_mode in ("all_background", "stable_background"):
            for method, features in METHOD_FEATURES.items():
                for feature in features:
                    selected = [
                        row
                        for row in gate_rows
                        if math.isfinite(float(row.get(feature, float("nan"))))
                        and (
                            row["label"] in {"parcial", "suited"}
                            or (
                                row["label"] == "background"
                                and (
                                    background_mode == "all_background"
                                    or row["stable_background"]
                                )
                            )
                        )
                    ]
                    labels = np.asarray(
                        [row["label"] in {"parcial", "suited"} for row in selected],
                        dtype=bool,
                    )
                    values = np.asarray([float(row[feature]) for row in selected])
                    if not labels.any() or labels.all() or not np.ptp(values):
                        continue
                    raw_auc = rank_auc(values[labels], values[~labels])
                    direction = 1.0 if raw_auc >= 0.5 else -1.0
                    directed = values * direction
                    point = operating_point(labels, directed)
                    output.append(
                        {
                            "gate_mode": gate_mode,
                            "background_mode": background_mode,
                            "method": method,
                            "feature": feature,
                            "n_positive": int(np.sum(labels)),
                            "n_negative": int(np.sum(~labels)),
                            "higher_raw_is_relevant": direction > 0,
                            "roc_auc": raw_auc if direction > 0 else 1.0 - raw_auc,
                            "pr_auc": average_precision(labels, directed),
                            **point,
                            "positive_median": float(np.median(values[labels])),
                            "negative_median": float(np.median(values[~labels])),
                        }
                    )
    return sorted(
        output,
        key=lambda row: (
            row["gate_mode"],
            row["background_mode"],
            row["method"],
            -row["roc_auc"],
        ),
    )


def best_method_features(performance: list[dict]) -> list[dict]:
    candidates = [
        row
        for row in performance
        if row["gate_mode"] == "oracle_label"
        and row["background_mode"] == "all_background"
    ]
    output = []
    for method in METHOD_FEATURES:
        method_rows = [row for row in candidates if row["method"] == method]
        output.append(max(method_rows, key=lambda row: (row["roc_auc"], row["pr_auc"])))
    return output


def operational_metrics(
    rows: list[dict],
    indexes: dict[str, list[dict]],
    best: list[dict],
) -> tuple[list[dict], list[dict]]:
    lookup = {
        (row["gate_mode"], row["passage_id"], row["capture_index"]): row
        for row in rows
    }
    summaries = []
    details = []
    for configuration in best:
        method = configuration["method"]
        feature = configuration["feature"]
        direction = 1.0 if configuration["higher_raw_is_relevant"] else -1.0
        threshold = float(configuration["threshold_directed"])
        for gate_mode in ("predicted_p99", "oracle_label"):
            current_details = []
            for passage_id, frames in indexes.items():
                state = False
                no_motion = 0
                pre_states = []
                post_states = []
                activation_indices = []
                for capture_index, frame in enumerate(frames, start=1):
                    row = lookup[(gate_mode, passage_id, capture_index)]
                    pre_states.append(state)
                    value = float(row.get(feature, float("nan")))
                    # INVALID e baseline nao sao observacoes de no-motion. Eles
                    # quebram o historico, mas preservam o estado corrente.
                    if math.isfinite(value):
                        moving = value * direction >= threshold
                        previous_state = state
                        if moving:
                            state = True
                            no_motion = 0
                        elif state:
                            no_motion += 1
                            if no_motion >= IDLE_PATIENCE:
                                state = False
                                no_motion = 0
                        if state and not previous_state:
                            activation_indices.append(capture_index - 1)
                    post_states.append(state)

                timestamps = np.asarray(
                    [float(frame["relative_time_ms"]) for frame in frames], dtype=float
                )
                labels = [frame["label"] for frame in frames]
                suited = [index for index, label in enumerate(labels) if label == "suited"]
                background = [
                    index for index, label in enumerate(labels) if label == "background"
                ]
                first_suited = suited[0] if suited else None
                first_activation = activation_indices[0] if activation_indices else None
                intervals = np.diff(timestamps)
                active_time = float(
                    np.sum(intervals[np.asarray(post_states[:-1], dtype=bool)])
                )
                total_time = float(np.sum(intervals))
                detail = {
                    "method": method,
                    "feature": feature,
                    "gate_mode": gate_mode,
                    "passage_id": passage_id,
                    "n_frames": len(frames),
                    "n_suited": len(suited),
                    "n_suited_forward_active": sum(pre_states[index] for index in suited),
                    "suited_passage_covered_forward": bool(
                        any(pre_states[index] for index in suited)
                    ),
                    "first_activation_before_suited": bool(
                        first_activation is not None
                        and first_suited is not None
                        and first_activation < first_suited
                    ),
                    "first_activation_in_partial": bool(
                        first_activation is not None
                        and labels[first_activation] == "parcial"
                    ),
                    "activation_delay_to_first_suited_ms": (
                        None
                        if first_activation is None or first_suited is None
                        else float(timestamps[first_activation] - timestamps[first_suited])
                    ),
                    "false_active_background_frames": sum(
                        post_states[index] for index in background
                    ),
                    "background_frames": len(background),
                    "time_active_ms": active_time,
                    "total_time_ms": total_time,
                    "time_active_ratio": active_time / total_time if total_time else 0.0,
                }
                current_details.append(detail)
                details.append(detail)

            suited_passages = [row for row in current_details if row["n_suited"]]
            delays = [
                row["activation_delay_to_first_suited_ms"]
                for row in suited_passages
                if row["activation_delay_to_first_suited_ms"] is not None
            ]
            total_suited = sum(row["n_suited"] for row in suited_passages)
            total_background = sum(row["background_frames"] for row in current_details)
            summaries.append(
                {
                    "method": method,
                    "feature": feature,
                    "gate_mode": gate_mode,
                    "threshold_raw": threshold * direction,
                    "threshold_direction": direction,
                    "idle_patience_frames": IDLE_PATIENCE,
                    "n_passages": len(current_details),
                    "n_suited_passages": len(suited_passages),
                    "suited_passage_coverage_forward": sum(
                        row["suited_passage_covered_forward"] for row in suited_passages
                    )
                    / len(suited_passages),
                    "suited_frame_retention_forward": sum(
                        row["n_suited_forward_active"] for row in suited_passages
                    )
                    / total_suited,
                    "passages_activated_before_first_suited": sum(
                        row["first_activation_before_suited"] for row in suited_passages
                    ),
                    "fraction_activated_before_first_suited": sum(
                        row["first_activation_before_suited"] for row in suited_passages
                    )
                    / len(suited_passages),
                    "passages_first_activation_in_partial": sum(
                        row["first_activation_in_partial"] for row in suited_passages
                    ),
                    "fraction_first_activation_in_partial": sum(
                        row["first_activation_in_partial"] for row in suited_passages
                    )
                    / len(suited_passages),
                    "activation_delay_median_ms": float(np.median(delays)) if delays else None,
                    "activation_delay_p95_ms": float(np.quantile(delays, 0.95)) if delays else None,
                    "false_active_background_ratio": sum(
                        row["false_active_background_frames"] for row in current_details
                    )
                    / total_background,
                    "time_active_ratio_global": sum(
                        row["time_active_ms"] for row in current_details
                    )
                    / sum(row["total_time_ms"] for row in current_details),
                    "time_active_ratio_median_by_passage": float(
                        np.median([row["time_active_ratio"] for row in current_details])
                    ),
                }
            )
    return summaries, details


def timing_summary(samples: dict[str, list[dict]]) -> list[dict]:
    output = []
    for method, rows in samples.items():
        for phase in ("preprocessing_ms", "algorithm_ms", "decision_ms", "total_ms"):
            summary = quantiles(float(row[phase]) for row in rows)
            output.append(
                {
                    "method": method,
                    "phase": phase.removesuffix("_ms"),
                    **summary,
                    "ops_per_s_from_mean_total": (
                        1000.0 / summary["mean"]
                        if phase == "total_ms" and summary["mean"]
                        else None
                    ),
                }
            )
    return output


def benchmark_methods(
    data_root: Path,
    indexes: dict[str, list[dict]],
    p99_lookup: dict[tuple[str, int], float],
    limit: int = 500,
) -> list[dict]:
    """Microbenchmark focado, com as tres fases separadas.

    Usa somente sequencias humanas validas e para ao atingir ``limit`` decisoes
    por metodo. Os valores servem apenas para ranking local.
    """

    samples: dict[str, list[dict]] = defaultdict(list)
    for passage_id, rows in indexes.items():
        frames = [
            base.read_depth(data_root / "DEPTH" / passage_id / row["depth_filename"])
            for row in rows
        ]
        valid_run: deque[np.ndarray] = deque(maxlen=6)
        mhi_masks: deque[np.ndarray] = deque(maxlen=3)
        for capture_index, (row, current) in enumerate(zip(rows, frames), start=1):
            if row["label"] == "ruido":
                valid_run.clear()
                mhi_masks.clear()
                continue
            if valid_run:
                previous = valid_run[-1]
                region = base.roi_slices(current.shape, ROI_B)

                start = time.perf_counter_ns()
                previous_roi = previous[region].astype(np.int32, copy=False)
                current_roi = current[region].astype(np.int32, copy=False)
                pre_end = time.perf_counter_ns()
                diff = np.abs(current_roi - previous_roi)
                mask = diff >= PIXEL_THRESHOLD_MM
                algorithm_end = time.perf_counter_ns()
                prior.component_geometry(mask)
                end = time.perf_counter_ns()
                samples["baseline_components"].append(
                    {
                        "preprocessing_ms": ns_to_ms(pre_end - start),
                        "algorithm_ms": ns_to_ms(algorithm_end - pre_end),
                        "decision_ms": ns_to_ms(end - algorithm_end),
                        "total_ms": ns_to_ms(end - start),
                    }
                )

                start = time.perf_counter_ns()
                previous_small = downsample_mean(previous[region].astype(np.float32)) / 2600.0
                current_small = downsample_mean(current[region].astype(np.float32)) / 2600.0
                pre_end = time.perf_counter_ns()
                flow = optical_flow_ilk(
                    previous_small,
                    current_small,
                    radius=FLOW_RADIUS,
                    num_warp=FLOW_NUM_WARP,
                    gaussian=True,
                    prefilter=True,
                    dtype=np.float32,
                )
                algorithm_end = time.perf_counter_ns()
                vertical, horizontal = flow
                magnitude = np.hypot(horizontal, vertical)
                _ = (
                    np.quantile(magnitude, 0.90)
                    * np.mean(np.abs(horizontal) / np.maximum(magnitude, 1e-6))
                )
                end = time.perf_counter_ns()
                samples["optical_flow_ilk"].append(
                    {
                        "preprocessing_ms": ns_to_ms(pre_end - start),
                        "algorithm_ms": ns_to_ms(algorithm_end - pre_end),
                        "decision_ms": ns_to_ms(end - algorithm_end),
                        "total_ms": ns_to_ms(end - start),
                    }
                )

                mhi_start = time.perf_counter_ns()
                mhi_mask = mask.astype(bool, copy=False)
                mhi_pre_end = time.perf_counter_ns()
                mhi_masks.append(mhi_mask)
                if len(mhi_masks) == 3:
                    counts = np.sum(np.asarray(mhi_masks, dtype=np.uint8), axis=0)
                    active = counts > 0
                else:
                    active = mhi_mask
                mhi_algorithm_end = time.perf_counter_ns()
                prior.component_geometry(active)
                mhi_end = time.perf_counter_ns()
                samples["motion_history"].append(
                    {
                        "preprocessing_ms": ns_to_ms(mhi_pre_end - mhi_start),
                        "algorithm_ms": ns_to_ms(mhi_algorithm_end - mhi_pre_end),
                        "decision_ms": ns_to_ms(mhi_end - mhi_algorithm_end),
                        "total_ms": ns_to_ms(mhi_end - mhi_start),
                    }
                )

            if len(valid_run) >= 2:
                start = time.perf_counter_ns()
                region = base.roi_slices(current.shape, ROI_B)
                older = valid_run[-2][region].astype(np.int32, copy=False)
                middle = valid_run[-1][region].astype(np.int32, copy=False)
                newer = current[region].astype(np.int32, copy=False)
                pre_end = time.perf_counter_ns()
                first = np.abs(middle - older) >= PIXEL_THRESHOLD_MM
                second = np.abs(newer - middle) >= PIXEL_THRESHOLD_MM
                union = first | second
                algorithm_end = time.perf_counter_ns()
                prior.component_geometry(union)
                end = time.perf_counter_ns()
                samples["three_frame_differencing"].append(
                    {
                        "preprocessing_ms": ns_to_ms(pre_end - start),
                        "algorithm_ms": ns_to_ms(algorithm_end - pre_end),
                        "decision_ms": ns_to_ms(end - algorithm_end),
                        "total_ms": ns_to_ms(end - start),
                    }
                )

            if len(valid_run) >= ADAPTIVE_WINDOW:
                start = time.perf_counter_ns()
                region = base.roi_slices(current.shape, ROI_B)
                stack = np.stack(
                    [frame[region] for frame in list(valid_run)[-ADAPTIVE_WINDOW:]]
                )
                current_roi = current[region].astype(np.float32, copy=False)
                pre_end = time.perf_counter_ns()
                baseline = np.median(stack, axis=0)
                adaptive_mask = (
                    np.abs(current_roi - baseline) >= PIXEL_THRESHOLD_MM
                )
                algorithm_end = time.perf_counter_ns()
                prior.component_geometry(adaptive_mask)
                end = time.perf_counter_ns()
                samples["adaptive_background"].append(
                    {
                        "preprocessing_ms": ns_to_ms(pre_end - start),
                        "algorithm_ms": ns_to_ms(algorithm_end - pre_end),
                        "decision_ms": ns_to_ms(end - algorithm_end),
                        "total_ms": ns_to_ms(end - start),
                    }
                )
            valid_run.append(current)
            if all(len(samples[method]) >= limit for method in METHOD_FEATURES):
                return timing_summary({method: values[:limit] for method, values in samples.items()})
    return timing_summary(samples)


def flow_distributions(rows: list[dict]) -> list[dict]:
    output = []
    flow_rows = [
        row
        for row in rows
        if row["gate_mode"] == "oracle_label"
        and math.isfinite(float(row.get("flow_p90_magnitude_px", float("nan"))))
    ]
    groups = {
        "relevant_partial_or_suited": lambda row: row["label"] in {"parcial", "suited"},
        "background": lambda row: row["label"] == "background",
        "stable_background": lambda row: row["label"] == "background"
        and row["stable_background"],
    }
    for group, predicate in groups.items():
        selected = [row for row in flow_rows if predicate(row)]
        for feature in (
            *METHOD_FEATURES["optical_flow_ilk"],
            "flow_signed_mean_horizontal_px",
            "flow_signed_mean_vertical_px",
            "flow_dominant_angle_degrees",
        ):
            output.append(
                {
                    "group": group,
                    "feature": feature,
                    **quantiles(float(row[feature]) for row in selected),
                }
            )
    return output


def flow_orientation_counts(rows: list[dict]) -> list[dict]:
    flow_rows = [
        row
        for row in rows
        if row["gate_mode"] == "oracle_label"
        and row.get("flow_dominant_orientation")
    ]
    groups = {
        "relevant_partial_or_suited": lambda row: row["label"] in {"parcial", "suited"},
        "background": lambda row: row["label"] == "background",
        "stable_background": lambda row: row["label"] == "background"
        and row["stable_background"],
    }
    output = []
    for group, predicate in groups.items():
        selected = [row for row in flow_rows if predicate(row)]
        for orientation in ("left", "right", "up", "down"):
            count = sum(row["flow_dominant_orientation"] == orientation for row in selected)
            output.append(
                {
                    "group": group,
                    "orientation": orientation,
                    "count": count,
                    "fraction": count / len(selected) if selected else None,
                }
            )
    return output


def flow_panel(
    data_root: Path,
    row: dict,
    output_path: Path,
    title: str,
) -> None:
    passage_id = row["passage_id"]
    current = base.read_depth(data_root / "DEPTH" / passage_id / row["depth_filename"])
    # O frame anterior e obtido pelo indice no simulation_index no chamador.
    previous = base.read_depth(
        data_root / "DEPTH" / passage_id / row["previous_depth_filename"]
    )
    features, flow = flow_features(previous, current)
    preview = base.depth_preview(current).convert("RGB")
    draw = ImageDraw.Draw(preview)
    y_slice, x_slice = base.roi_slices(current.shape, ROI_B)
    draw.rectangle(
        (x_slice.start, y_slice.start, x_slice.stop - 1, y_slice.stop - 1),
        outline=(255, 255, 0),
        width=2,
    )
    vertical, horizontal = flow
    stride = 6
    for y in range(stride // 2, flow.shape[1], stride):
        for x in range(stride // 2, flow.shape[2], stride):
            u = float(horizontal[y, x])
            v = float(vertical[y, x])
            if math.hypot(u, v) < FLOW_MAGNITUDE_MIN_PX:
                continue
            px = x_slice.start + x * FLOW_DOWNSAMPLE
            py = y_slice.start + y * FLOW_DOWNSAMPLE
            scale = 1.5 * FLOW_DOWNSAMPLE
            draw.line((px, py, px + u * scale, py + v * scale), fill=(255, 0, 0), width=1)
    panel = Image.new("RGB", (320, 285), "white")
    panel.paste(preview, (0, 45))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((5, 4), title, fill="black")
    panel_draw.text(
        (5, 22),
        f"p90={features['flow_p90_magnitude_px']:.2f}px "
        f"dir={features['flow_directional_coherence']:.2f} "
        f"h-sign={features['flow_horizontal_sign_coherence']:.2f}",
        fill="black",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def create_flow_panels(
    data_root: Path,
    indexes: dict[str, list[dict]],
    rows: list[dict],
    best_flow: dict,
    output_dir: Path,
) -> list[dict]:
    feature = best_flow["feature"]
    direction = 1.0 if best_flow["higher_raw_is_relevant"] else -1.0
    selected_rows = [
        row
        for row in rows
        if row["gate_mode"] == "oracle_label"
        and math.isfinite(float(row.get(feature, float("nan"))))
    ]
    manifest = []
    for group, predicate in (
        ("entry_relevant", lambda row: row["label"] in {"parcial", "suited"}),
        ("background_false_positive", lambda row: row["label"] == "background"),
    ):
        candidates = sorted(
            [row for row in selected_rows if predicate(row)],
            key=lambda row: float(row[feature]) * direction,
            reverse=True,
        )[:6]
        for rank, row in enumerate(candidates, start=1):
            passage_rows = indexes[row["passage_id"]]
            row = dict(row)
            row["previous_depth_filename"] = passage_rows[int(row["capture_index"]) - 2][
                "depth_filename"
            ]
            path = output_dir / "flow_examples" / f"{group}_{rank:02d}.png"
            flow_panel(
                data_root,
                row,
                path,
                f"{group} {row['passage_id']} #{row['capture_index']} {row['label']}",
            )
            manifest.append(
                {
                    "group": group,
                    "rank": rank,
                    "passage_id": row["passage_id"],
                    "capture_index": row["capture_index"],
                    "label": row["label"],
                    "feature": feature,
                    "feature_value": row[feature],
                    "path": str(path.relative_to(REPO_ROOT)),
                }
            )
    return manifest


def create_summary_chart(
    performance: list[dict], timings: list[dict], output_path: Path
) -> None:
    best = best_method_features(performance)
    latency = {
        row["method"]: float(row["mean"])
        for row in timings
        if row["phase"] == "total"
    }
    width, height = 980, 430
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 12), "Classical PDI: oracle-valid ROC-AUC and local mean latency", fill="black")
    left, top, plot_w, plot_h = 70, 55, 850, 285
    draw.line((left, top, left, top + plot_h), fill="black")
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black")
    colors = [(70, 110, 180), (215, 125, 45), (70, 155, 90), (170, 80, 160), (90, 150, 160)]
    bar_width = 90
    gap = 65
    for index, row in enumerate(best):
        x = left + 35 + index * (bar_width + gap)
        auc = float(row["roc_auc"])
        bar_h = auc * plot_h
        draw.rectangle((x, top + plot_h - bar_h, x + bar_width, top + plot_h), fill=colors[index])
        draw.text((x + 17, top + plot_h - bar_h - 18), f"{auc:.3f}", fill="black")
        label = row["method"].replace("_", " ")
        draw.text((x - 15, top + plot_h + 8), label[:18], fill="black")
        draw.text((x + 5, top + plot_h + 27), f"{latency.get(row['method'], math.nan):.2f} ms", fill="black")
    draw.text((20, top + 125), "ROC-AUC", fill="black")
    draw.text((left + 265, height - 25), "Bars: AUC; text below: observed local mean feature cost", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
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
    p99_lookup = load_frame_p99(args.frame_features)
    if len(p99_lookup) != EXPECTED_COHORT[1]:
        raise ValueError(f"frame P99 lookup mismatch: {len(p99_lookup)}")

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, timing_samples, quality_summary = extract_features(
        args.data_root, indexes, p99_lookup
    )
    performance = feature_performance(rows)
    best = best_method_features(performance)
    operational, passage_details = operational_metrics(rows, indexes, best)
    # O benchmark dedicado decompõe custo; os timings da extração completa são
    # mantidos apenas para auditoria no summary JSON.
    timings = benchmark_methods(args.data_root, indexes, p99_lookup)
    flow_distribution = flow_distributions(rows)
    flow_orientations = flow_orientation_counts(rows)
    best_flow = next(row for row in best if row["method"] == "optical_flow_ilk")
    flow_panels = create_flow_panels(
        args.data_root, indexes, rows, best_flow, args.output
    )

    write_csv(args.output / "method_frame_performance.csv", performance)
    write_csv(args.output / "method_best_features.csv", best)
    write_csv(args.output / "method_operational_summary.csv", operational)
    write_csv(args.output / "method_operational_by_passage.csv", passage_details)
    write_csv(args.output / "method_latency.csv", timings)
    write_csv(args.output / "optical_flow_distributions.csv", flow_distribution)
    write_csv(args.output / "optical_flow_orientation_counts.csv", flow_orientations)
    write_csv(args.output / "optical_flow_examples.csv", flow_panels)
    # Feature table is useful for independent verification but omits arrays.
    metadata_fields = [
        "gate_mode",
        "passage_id",
        "capture_index",
        "relative_time_ms",
        "label",
        "depth_filename",
        "depth_p99_mm",
        "predicted_invalid",
        "oracle_invalid",
        "stable_background",
        "has_temporal_score",
    ]
    all_fields = set().union(*(row.keys() for row in rows))
    feature_fields = sorted(all_fields.difference(metadata_fields))
    write_csv(
        args.output / "temporal_feature_rows.csv",
        rows,
        fieldnames=metadata_fields + feature_fields,
    )
    create_summary_chart(
        performance, timings, args.output / "method_quality_latency_summary.png"
    )

    summary = {
        "scope": "offline classical PDI audit; no runtime or model changes",
        "cohort": {
            "passages": observed[0],
            "frames": observed[1],
            "suited_frames": observed[2],
        },
        "quality_gate": {
            "runtime_candidate": "depth_p99_mm >= 2230",
            "threshold_mm": QUALITY_P99_THRESHOLD_MM,
            "reset_semantics": (
                "invalid clears temporal history; next valid is baseline without score; "
                "no comparison crosses invalid"
            ),
            "state_semantics_during_invalid": (
                "invalid/baseline do not count as no-motion and preserve current state"
            ),
            **quality_summary,
        },
        "baseline": {
            "roi": "y30-70%, x20-80%",
            "pixel_threshold_mm": PIXEL_THRESHOLD_MM,
            "score": "largest_component_area / changed_pixels",
        },
        "optical_flow": {
            "implementation": "skimage.registration.optical_flow_ilk",
            "roi": "ROI B",
            "input_shape": [48, 96],
            "downsample": FLOW_DOWNSAMPLE,
            "radius": FLOW_RADIUS,
            "num_warp": FLOW_NUM_WARP,
        },
        "mhi_windows": list(MHI_WINDOWS),
        "adaptive_background_window": ADAPTIVE_WINDOW,
        "stable_background_definition": (
            "current frame is the center of five consecutive human background labels"
        ),
        "operational_policy": {
            "idle_patience_frames": IDLE_PATIENCE,
            "threshold": "Youden point learned on oracle-valid all-background comparison",
            "forward_looking": "state before observing/acquiring the current frame",
            "time_interval_assignment": "post-observation state owns [t_i, t_i+1)",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "full_extraction_total_latency_ms": {
            row["method"]: row["mean"]
            for row in timing_summary(timing_samples)
            if row["phase"] == "total"
        },
    }
    (args.output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
