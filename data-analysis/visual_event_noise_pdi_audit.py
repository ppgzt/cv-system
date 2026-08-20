#!/usr/bin/env python3
"""Auditoria offline de ruido e sinais PDI para o Visual Event.

Somente le o cohort operacional e gera artefatos isolados. Nao executa PADE,
modelos ou codigo do pipeline e nao altera configuracoes do detector online.
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
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, stats


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import visual_event_diagnostic as base  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "data-analysis/visual_event_noise_pdi_output"
HISTOGRAM_BINS = 64
PIXEL_THRESHOLDS_MM = (100.0, 200.0)
LABELS = ("background", "parcial", "ruido", "suited")
ROI_DEFINITIONS = {
    "roi_a_y25_75_x20_80": (0.25, 0.75, 0.20, 0.80),
    "roi_b_y30_70_x20_80": (0.30, 0.70, 0.20, 0.80),
    "roi_c_y20_75_x20_80": (0.20, 0.75, 0.20, 0.80),
    "roi_d_y25_70_x20_80": (0.25, 0.70, 0.20, 0.80),
    "roi_e_y25_75_x15_85": (0.25, 0.75, 0.15, 0.85),
}
COMPONENT_ROIS = {"roi_a_y25_75_x20_80", "roi_b_y30_70_x20_80"}
FRAME_METADATA = {
    "passage_id",
    "capture_index",
    "relative_time_ms",
    "passage_start_timestamp_ms",
    "elapsed_from_passage_start_ms",
    "label",
    "depth_filename",
}
PAIR_METADATA = {
    "passage_id",
    "capture_index",
    "previous_label",
    "label",
    "transition",
    "previous_relative_time_ms",
    "relative_time_ms",
    "delta_t_ms",
    "previous_depth_filename",
    "depth_filename",
}


def finite_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def frame_features(frame: np.ndarray) -> dict:
    flat = frame.reshape(-1).astype(np.float32, copy=False)
    quantiles = np.quantile(flat, (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99))
    hist, _ = np.histogram(flat, bins=HISTOGRAM_BINS, range=(0.0, 2600.0))
    probabilities = hist[hist > 0].astype(np.float64) / flat.size
    dx = np.abs(np.diff(frame.astype(np.int32, copy=False), axis=1)).reshape(-1)
    dy = np.abs(np.diff(frame.astype(np.int32, copy=False), axis=0)).reshape(-1)
    gradients = np.concatenate((dx, dy)).astype(np.float32, copy=False)
    median = float(quantiles[3])
    return {
        "depth_mean_mm": float(np.mean(flat)),
        "depth_median_mm": median,
        "depth_std_mm": float(np.std(flat)),
        "depth_p1_mm": float(quantiles[0]),
        "depth_p5_mm": float(quantiles[1]),
        "depth_p25_mm": float(quantiles[2]),
        "depth_p75_mm": float(quantiles[4]),
        "depth_p95_mm": float(quantiles[5]),
        "depth_p99_mm": float(quantiles[6]),
        "depth_p95_minus_p5_mm": float(quantiles[5] - quantiles[1]),
        "depth_iqr_mm": float(quantiles[4] - quantiles[2]),
        "depth_mad_from_median_mm": float(np.median(np.abs(flat - median))),
        "histogram_entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))),
        "histogram_occupied_bin_fraction": float(np.count_nonzero(hist) / HISTOGRAM_BINS),
        "histogram_dominant_bin_fraction": float(np.max(hist) / flat.size),
        "fraction_le_50mm": float(np.mean(flat <= 50.0)),
        "fraction_le_100mm": float(np.mean(flat <= 100.0)),
        "fraction_ge_2500mm": float(np.mean(flat >= 2500.0)),
        "fraction_equal_2600mm": float(np.mean(flat == 2600.0)),
        "gradient_mean_mm": float(np.mean(gradients)),
        "gradient_median_mm": float(np.median(gradients)),
        "gradient_p95_mm": float(np.quantile(gradients, 0.95)),
    }


def collect_frame_features(data_root: Path, indexes: dict[str, list[dict]]) -> list[dict]:
    output = []
    for passage_id, rows in indexes.items():
        passage_start = float(rows[0]["relative_time_ms"])
        for capture_index, row in enumerate(rows, start=1):
            frame = base.read_depth(data_root / "DEPTH" / passage_id / row["depth_filename"])
            output.append(
                {
                    "passage_id": passage_id,
                    "capture_index": capture_index,
                    "relative_time_ms": float(row["relative_time_ms"]),
                    "passage_start_timestamp_ms": passage_start,
                    "elapsed_from_passage_start_ms": float(row["relative_time_ms"])
                    - passage_start,
                    "label": row["label"],
                    "depth_filename": row["depth_filename"],
                    **frame_features(frame),
                }
            )
    return output


def label_temporal_audit(frame_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    by_index = defaultdict(Counter)
    for row in frame_rows:
        by_index[row["capture_index"]][row["label"]] += 1
    index_rows = []
    for capture_index in sorted(by_index):
        total = sum(by_index[capture_index].values())
        for label in LABELS:
            index_rows.append(
                {
                    "capture_index": capture_index,
                    "label": label,
                    "count": by_index[capture_index][label],
                    "fraction_at_capture_index": by_index[capture_index][label] / total,
                }
            )

    summary = []
    for label in LABELS:
        selected = [row for row in frame_rows if row["label"] == label]
        for limit in (1, 2, 3, 5, 10):
            count = sum(row["capture_index"] <= limit for row in selected)
            summary.append(
                {
                    "label": label,
                    "window_type": "first_source_frames",
                    "window_limit": limit,
                    "window_unit": "capture_index",
                    "count": count,
                    "total_label_frames": len(selected),
                    "cumulative_fraction": count / len(selected) if selected else None,
                }
            )
        for limit in (100, 200, 300, 500, 1000):
            count = sum(
                row["elapsed_from_passage_start_ms"] <= limit for row in selected
            )
            summary.append(
                {
                    "label": label,
                    "window_type": "elapsed_from_passage_start",
                    "window_limit": limit,
                    "window_unit": "ms",
                    "count": count,
                    "total_label_frames": len(selected),
                    "cumulative_fraction": count / len(selected) if selected else None,
                }
            )

    passage_rows = []
    by_passage = defaultdict(list)
    for row in frame_rows:
        by_passage[row["passage_id"]].append(row)
    for passage_id, rows in sorted(by_passage.items()):
        for label in LABELS:
            selected = [row for row in rows if row["label"] == label]
            passage_rows.append(
                {
                    "passage_id": passage_id,
                    "label": label,
                    "count": len(selected),
                    "has_after_capture_index_3": any(
                        row["capture_index"] > 3 for row in selected
                    ),
                    "has_after_500ms": any(
                        row["elapsed_from_passage_start_ms"] > 500 for row in selected
                    ),
                    "last_capture_index": max(
                        (row["capture_index"] for row in selected), default=None
                    ),
                    "last_elapsed_from_passage_start_ms": max(
                        (
                            row["elapsed_from_passage_start_ms"]
                            for row in selected
                        ),
                        default=None,
                    ),
                }
            )
    return index_rows, summary, passage_rows


def component_geometry(mask: np.ndarray) -> dict:
    labels, n_components = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if not n_components:
        return {
            "component_count": 0,
            "relevant_component_count": 0,
            "largest_component_area_px": 0,
            "largest_component_area_ratio": 0.0,
            "largest_component_changed_fraction": 0.0,
            "largest_bbox_width_px": 0,
            "largest_bbox_height_px": 0,
            "largest_bbox_width_ratio": 0.0,
            "largest_bbox_height_ratio": 0.0,
            "largest_bbox_aspect_ratio": None,
            "largest_bbox_occupancy": 0.0,
            "largest_centroid_x_ratio": None,
            "largest_centroid_y_ratio": None,
            "largest_touches_border": False,
            "largest_bbox_x0": None,
            "largest_bbox_y0": None,
            "largest_bbox_x1": None,
            "largest_bbox_y1": None,
        }
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(np.argmax(sizes)) + 1
    area = int(sizes[largest - 1])
    ys, xs = np.nonzero(labels == largest)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    changed = int(np.count_nonzero(mask))
    relevant_minimum = max(8, int(round(mask.size * 0.001)))
    return {
        "component_count": int(n_components),
        "relevant_component_count": int(np.count_nonzero(sizes >= relevant_minimum)),
        "largest_component_area_px": area,
        "largest_component_area_ratio": area / mask.size,
        "largest_component_changed_fraction": area / changed if changed else 0.0,
        "largest_bbox_width_px": width,
        "largest_bbox_height_px": height,
        "largest_bbox_width_ratio": width / mask.shape[1],
        "largest_bbox_height_ratio": height / mask.shape[0],
        "largest_bbox_aspect_ratio": width / height if height else None,
        "largest_bbox_occupancy": area / (width * height),
        "largest_centroid_x_ratio": float(xs.mean() / mask.shape[1]),
        "largest_centroid_y_ratio": float(ys.mean() / mask.shape[0]),
        "largest_touches_border": bool(
            x0 == 0 or y0 == 0 or x1 == mask.shape[1] or y1 == mask.shape[0]
        ),
        "largest_bbox_x0": x0,
        "largest_bbox_y0": y0,
        "largest_bbox_x1": x1,
        "largest_bbox_y1": y1,
    }


def prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}__{key}": value for key, value in values.items()}


def robust_diff_features(diff: np.ndarray) -> dict:
    values = diff.reshape(-1)
    q25, median, q75, q90 = np.quantile(values, (0.25, 0.50, 0.75, 0.90))
    low, high = np.quantile(values, (0.10, 0.90))
    trimmed = values[(values >= low) & (values <= high)]
    return {
        "mad": float(np.mean(values)),
        "diff_median_mm": float(median),
        "diff_p75_mm": float(q75),
        "diff_p90_mm": float(q90),
        "diff_iqr_mm": float(q75 - q25),
        "diff_trimmed_mean_10_90_mm": float(np.mean(trimmed)) if trimmed.size else 0.0,
    }


def collect_pair_features(data_root: Path, indexes: dict[str, list[dict]]) -> list[dict]:
    output = []
    for passage_id, rows in indexes.items():
        previous = None
        previous_row = None
        for capture_index, row in enumerate(rows, start=1):
            current = base.read_depth(data_root / "DEPTH" / passage_id / row["depth_filename"])
            if previous is not None:
                diff = np.abs(
                    current.astype(np.int32, copy=False)
                    - previous.astype(np.int32, copy=False)
                ).astype(np.float32)
                pair = {
                    "passage_id": passage_id,
                    "capture_index": capture_index,
                    "previous_label": previous_row["label"],
                    "label": row["label"],
                    "transition": f"{previous_row['label']}->{row['label']}",
                    "previous_relative_time_ms": float(previous_row["relative_time_ms"]),
                    "relative_time_ms": float(row["relative_time_ms"]),
                    "delta_t_ms": float(row["relative_time_ms"])
                    - float(previous_row["relative_time_ms"]),
                    "previous_depth_filename": previous_row["depth_filename"],
                    "depth_filename": row["depth_filename"],
                    **prefixed("global", robust_diff_features(diff)),
                }
                for threshold in PIXEL_THRESHOLDS_MM:
                    mask = diff >= threshold
                    pair[f"global__changed_ratio_{int(threshold)}mm"] = float(mask.mean())
                    pair.update(
                        prefixed(
                            f"global_{int(threshold)}mm",
                            component_geometry(mask),
                        )
                    )
                for roi_name, fractions in ROI_DEFINITIONS.items():
                    slices = base.roi_slices(diff.shape, fractions)
                    roi_diff = diff[slices]
                    pair.update(prefixed(roi_name, robust_diff_features(roi_diff)))
                    for threshold in PIXEL_THRESHOLDS_MM:
                        roi_mask = roi_diff >= threshold
                        pair[f"{roi_name}__changed_ratio_{int(threshold)}mm"] = float(
                            roi_mask.mean()
                        )
                        if roi_name in COMPONENT_ROIS:
                            geometry = component_geometry(roi_mask)
                            pair.update(
                                prefixed(
                                    f"{roi_name}_{int(threshold)}mm",
                                    geometry,
                                )
                            )
                            pair[
                                f"{roi_name}_{int(threshold)}mm__area_x_occupancy"
                            ] = (
                                geometry["largest_component_area_ratio"]
                                * geometry["largest_bbox_occupancy"]
                            )
                            centroid_y = geometry["largest_centroid_y_ratio"]
                            center_weight = (
                                0.0
                                if centroid_y is None
                                else max(0.0, 1.0 - abs(centroid_y - 0.5) * 2.0)
                            )
                            pair[
                                f"{roi_name}_{int(threshold)}mm__area_x_vertical_center"
                            ] = geometry["largest_component_area_ratio"] * center_weight
                output.append(pair)
            previous = current
            previous_row = row
    return output


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
    ranks = np.arange(1, len(labels) + 1)
    precision = true_positive / ranks
    positives = int(labels.sum())
    return float(precision[ordered].sum() / positives) if positives else 0.0


def operating_points(labels: np.ndarray, scores: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(~labels_sorted)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    tpr = tp / positives
    fpr = fp / negatives
    precision = tp / np.arange(1, len(labels) + 1)
    distinct = np.r_[scores_sorted[1:] != scores_sorted[:-1], True]
    indices = np.flatnonzero(distinct)

    youden_index = indices[np.argmax((tpr - fpr)[indices])]
    recall_candidates = indices[tpr[indices] >= 0.95]
    recall_index = (
        recall_candidates[np.argmin(fpr[recall_candidates])]
        if recall_candidates.size
        else indices[-1]
    )
    return {
        "youden_threshold_directed": float(scores_sorted[youden_index]),
        "youden_recall": float(tpr[youden_index]),
        "youden_fpr": float(fpr[youden_index]),
        "youden_precision": float(precision[youden_index]),
        "recall95_threshold_directed": float(scores_sorted[recall_index]),
        "recall95_recall": float(tpr[recall_index]),
        "recall95_fpr": float(fpr[recall_index]),
        "recall95_precision": float(precision[recall_index]),
    }


def feature_performance(
    rows: list[dict],
    feature_names: Iterable[str],
    positive: Callable[[dict], bool],
    negative: Callable[[dict], bool],
    evaluation: str,
) -> list[dict]:
    selected = [row for row in rows if positive(row) or negative(row)]
    labels = np.asarray([positive(row) for row in selected], dtype=bool)
    output = []
    for feature in feature_names:
        values = np.asarray([float(row[feature]) for row in selected], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.ptp(values) == 0:
            continue
        raw_auc = rank_auc(values[labels], values[~labels])
        direction = 1.0 if raw_auc >= 0.5 else -1.0
        directed = values * direction
        auc = raw_auc if direction > 0 else 1.0 - raw_auc
        points = operating_points(labels, directed)
        output.append(
            {
                "evaluation": evaluation,
                "feature": feature,
                "positive_count": int(labels.sum()),
                "negative_count": int((~labels).sum()),
                "positive_prevalence": float(labels.mean()),
                "higher_raw_value_is_positive": direction > 0,
                "roc_auc": auc,
                "pr_auc": average_precision(labels, directed),
                **points,
                "positive_median": float(np.median(values[labels])),
                "positive_p95": float(np.quantile(values[labels], 0.95)),
                "negative_median": float(np.median(values[~labels])),
                "negative_p95": float(np.quantile(values[~labels], 0.95)),
            }
        )
    return sorted(output, key=lambda row: (-row["roc_auc"], -row["pr_auc"]))


def feature_distributions(
    rows: list[dict], feature_names: Iterable[str], class_field: str
) -> list[dict]:
    output = []
    for feature in feature_names:
        for label in LABELS:
            values = [float(row[feature]) for row in rows if row[class_field] == label]
            output.append(
                {
                    "feature": feature,
                    "label": label,
                    **base.quantile_summary(values),
                }
            )
    return output


def histogram_rows(
    rows: list[dict], feature_names: Iterable[str], bins: int = 50
) -> list[dict]:
    output = []
    for feature in feature_names:
        all_values = np.asarray([float(row[feature]) for row in rows])
        low, high = np.quantile(all_values, (0.001, 0.999))
        if low == high:
            continue
        edges = np.linspace(low, high, bins + 1)
        for label in LABELS:
            values = np.asarray([float(row[feature]) for row in rows if row["label"] == label])
            counts, _ = np.histogram(values, bins=edges)
            for index, count in enumerate(counts):
                output.append(
                    {
                        "feature": feature,
                        "label": label,
                        "bin_index": index,
                        "bin_left": float(edges[index]),
                        "bin_right": float(edges[index + 1]),
                        "count": int(count),
                        "density_fraction": float(count / len(values)),
                    }
                )
    return output


def save_histogram_images(histogram_data: list[dict], output_dir: Path) -> list[dict]:
    image_dir = output_dir / "frame_feature_histograms"
    image_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "background": (80, 110, 180),
        "parcial": (230, 150, 45),
        "suited": (45, 150, 85),
        "ruido": (190, 55, 60),
    }
    metadata = []
    for feature in sorted({row["feature"] for row in histogram_data}):
        rows = [row for row in histogram_data if row["feature"] == feature]
        if not rows:
            continue
        width, height = 720, 340
        left, top, right, bottom = 60, 30, 20, 55
        plot_width = width - left - right
        plot_height = height - top - bottom
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.line((left, top, left, top + plot_height), fill="black", width=1)
        draw.line(
            (left, top + plot_height, left + plot_width, top + plot_height),
            fill="black",
            width=1,
        )
        maximum = max(float(row["density_fraction"]) for row in rows) or 1.0
        n_bins = max(int(row["bin_index"]) for row in rows) + 1
        for label in LABELS:
            label_rows = sorted(
                [row for row in rows if row["label"] == label],
                key=lambda row: int(row["bin_index"]),
            )
            points = []
            for row in label_rows:
                x = left + (int(row["bin_index"]) + 0.5) / n_bins * plot_width
                y = top + plot_height - float(row["density_fraction"]) / maximum * plot_height
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=colors[label], width=2)
        x_min = min(float(row["bin_left"]) for row in rows)
        x_max = max(float(row["bin_right"]) for row in rows)
        draw.text((left, 7), feature, fill="black")
        draw.text((left, top + plot_height + 8), f"{x_min:.3g}", fill="black")
        draw.text((left + plot_width - 55, top + plot_height + 8), f"{x_max:.3g}", fill="black")
        legend_x = left + 120
        for index, label in enumerate(LABELS):
            x = legend_x + index * 125
            draw.line((x, height - 18, x + 18, height - 18), fill=colors[label], width=3)
            draw.text((x + 23, height - 25), label, fill="black")
        safe = feature.replace("__", "_").replace("/", "_")
        path = image_dir / f"{safe}.png"
        image.save(path)
        metadata.append(
            {"feature": feature, "path": str(path.resolve().relative_to(REPO_ROOT))}
        )
    return metadata


def noise_subgroup_performance(
    frame_rows: list[dict], best: dict
) -> list[dict]:
    feature = best["feature"]
    direction = 1.0 if best["higher_raw_value_is_positive"] else -1.0
    thresholds = {
        "youden": best["youden_threshold_directed"],
        "recall95": best["recall95_threshold_directed"],
    }
    groups = {
        "noise_capture_index_le_3": lambda row: row["label"] == "ruido"
        and row["capture_index"] <= 3,
        "noise_capture_index_gt_3": lambda row: row["label"] == "ruido"
        and row["capture_index"] > 3,
        "noise_time_le_500ms": lambda row: row["label"] == "ruido"
        and row["elapsed_from_passage_start_ms"] <= 500,
        "noise_time_gt_500ms": lambda row: row["label"] == "ruido"
        and row["elapsed_from_passage_start_ms"] > 500,
        "non_noise": lambda row: row["label"] != "ruido",
    }
    output = []
    for threshold_name, threshold in thresholds.items():
        for group, predicate in groups.items():
            selected = [row for row in frame_rows if predicate(row)]
            predicted = [float(row[feature]) * direction >= threshold for row in selected]
            output.append(
                {
                    "feature": feature,
                    "threshold_kind": threshold_name,
                    "threshold_directed": threshold,
                    "group": group,
                    "n_frames": len(selected),
                    "predicted_noise_fraction": float(np.mean(predicted)) if selected else None,
                    **base.quantile_summary(float(row[feature]) for row in selected),
                }
            )
    return output


def movement_after_predicted_quality_gate(
    frame_rows: list[dict],
    pair_rows: list[dict],
    noise_performance: list[dict],
    movement_features: list[str],
) -> list[dict]:
    frame_lookup = {
        (row["passage_id"], row["capture_index"]): row for row in frame_rows
    }
    output = []
    for gate_feature in ("depth_p99_mm", "fraction_ge_2500mm"):
        gate = next(row for row in noise_performance if row["feature"] == gate_feature)
        direction = 1.0 if gate["higher_raw_value_is_positive"] else -1.0
        for threshold_kind in ("youden", "recall95"):
            threshold = gate[f"{threshold_kind}_threshold_directed"]
            filtered = []
            for pair in pair_rows:
                current = frame_lookup[(pair["passage_id"], pair["capture_index"])]
                previous = frame_lookup[(pair["passage_id"], pair["capture_index"] - 1)]
                current_invalid = float(current[gate_feature]) * direction >= threshold
                previous_invalid = float(previous[gate_feature]) * direction >= threshold
                if not current_invalid and not previous_invalid:
                    filtered.append(pair)
            performance = feature_performance(
                filtered,
                movement_features,
                lambda row: row["label"] in {"parcial", "suited"},
                lambda row: row["label"] == "background",
                "predicted_quality_gate_current_relevant_vs_background",
            )
            for row in performance:
                output.append(
                    {
                        "quality_gate_feature": gate_feature,
                        "quality_gate_threshold_kind": threshold_kind,
                        "quality_gate_threshold_directed": threshold,
                        "n_pairs_after_gate": len(filtered),
                        **row,
                    }
                )
    return output


def add_temporal_persistence(
    pair_rows: list[dict], feature_names: list[str]
) -> tuple[list[str], list[str]]:
    raw_features = []
    oracle_features = []
    by_passage = defaultdict(list)
    for row in pair_rows:
        by_passage[row["passage_id"]].append(row)
    for feature in feature_names:
        for length in (2, 3):
            new_feature = f"{feature}__persistence_min{length}"
            oracle_feature = f"{feature}__oracle_valid_persistence_min{length}"
            raw_features.append(new_feature)
            oracle_features.append(oracle_feature)
            for passage_rows in by_passage.values():
                raw_history = deque(maxlen=length)
                oracle_history = deque(maxlen=length)
                for row in sorted(passage_rows, key=lambda item: item["capture_index"]):
                    value = float(row[feature])
                    raw_history.append(value)
                    row[new_feature] = (
                        min(raw_history) if len(raw_history) == length else float("nan")
                    )
                    if row["previous_label"] == "ruido" or row["label"] == "ruido":
                        oracle_history.clear()
                        row[oracle_feature] = float("nan")
                        continue
                    oracle_history.append(value)
                    row[oracle_feature] = (
                        min(oracle_history)
                        if len(oracle_history) == length
                        else float("nan")
                    )
    return raw_features, oracle_features


def persistence_performance(
    pair_rows: list[dict], feature_names: list[str], evaluation: str
) -> list[dict]:
    output = []
    for feature in feature_names:
        selected = [
            row
            for row in pair_rows
            if row["previous_label"] != "ruido"
            and row["label"] != "ruido"
            and math.isfinite(float(row[feature]))
            and row["label"] in {"background", "parcial", "suited"}
        ]
        output.extend(
            feature_performance(
                selected,
                [feature],
                lambda row: row["label"] in {"parcial", "suited"},
                lambda row: row["label"] == "background",
                evaluation,
            )
        )
    return output


def passage_activation_metrics(
    indexes: dict[str, list[dict]],
    pair_rows: list[dict],
    configurations: list[dict],
) -> tuple[list[dict], list[dict]]:
    pair_lookup = {
        (row["passage_id"], row["capture_index"]): row for row in pair_rows
    }
    summaries = []
    details = []
    for configuration in configurations:
        feature = configuration["feature"]
        direction = 1.0 if configuration["higher_raw_value_is_positive"] else -1.0
        threshold = configuration["youden_threshold_directed"]
        quality_modes = (
            ("oracle_valid_pair",)
            if "oracle_valid_persistence" in feature
            else ("complete", "oracle_valid_pair")
        )
        for quality_mode in quality_modes:
            current_details = []
            for passage_id, frames in indexes.items():
                state = False
                no_motion = 0
                idle_patience = 3
                post_states = [False]
                pre_states = [False]
                activation_indices = []
                for capture_index in range(2, len(frames) + 1):
                    pair = pair_lookup[(passage_id, capture_index)]
                    pre_states.append(state)
                    if quality_mode == "oracle_valid_pair" and (
                        pair["previous_label"] == "ruido" or pair["label"] == "ruido"
                    ):
                        state = False
                        no_motion = 0
                        post_states.append(state)
                        continue
                    value = float(pair.get(feature, float("nan")))
                    moving = math.isfinite(value) and value * direction >= threshold
                    previous_state = state
                    if moving:
                        state = True
                        no_motion = 0
                    elif state:
                        no_motion += 1
                        if no_motion >= idle_patience:
                            state = False
                            no_motion = 0
                    if state and not previous_state:
                        activation_indices.append(capture_index - 1)
                    post_states.append(state)

                timestamps = np.asarray(
                    [float(frame["relative_time_ms"]) for frame in frames], dtype=float
                )
                intervals = np.diff(timestamps)
                labels = [frame["label"] for frame in frames]
                suited_indices = [index for index, label in enumerate(labels) if label == "suited"]
                first_suited_index = suited_indices[0] if suited_indices else None
                first_activation_index = activation_indices[0] if activation_indices else None
                active_time = float(
                    intervals[np.asarray(post_states[:-1], dtype=bool)].sum()
                )
                total_time = float(intervals.sum())
                suited_post = sum(post_states[index] for index in suited_indices)
                suited_forward = sum(pre_states[index] for index in suited_indices)
                background_indices = [
                    index for index, label in enumerate(labels) if label == "background"
                ]
                detail = {
                    "feature": feature,
                    "quality_mode": quality_mode,
                    "threshold_kind": "youden_valid_pair",
                    "threshold_directed": threshold,
                    "idle_patience_frames": idle_patience,
                    "passage_id": passage_id,
                    "n_frames": len(frames),
                    "n_suited": len(suited_indices),
                    "n_suited_post_active": suited_post,
                    "n_suited_forward_active": suited_forward,
                    "suited_passage_covered_post": bool(suited_post),
                    "suited_passage_covered_forward": bool(suited_forward),
                    "first_suited_timestamp_ms": None
                    if first_suited_index is None
                    else timestamps[first_suited_index],
                    "first_activation_timestamp_ms": None
                    if first_activation_index is None
                    else timestamps[first_activation_index],
                    "activation_delay_to_first_suited_ms": None
                    if first_suited_index is None or first_activation_index is None
                    else timestamps[first_activation_index] - timestamps[first_suited_index],
                    "first_activation_before_suited": bool(
                        first_suited_index is not None
                        and first_activation_index is not None
                        and first_activation_index < first_suited_index
                    ),
                    "first_activation_in_partial": bool(
                        first_activation_index is not None
                        and labels[first_activation_index] == "parcial"
                    ),
                    "false_active_background_frames": sum(
                        post_states[index] for index in background_indices
                    ),
                    "background_frames": len(background_indices),
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
            total_suited = sum(row["n_suited"] for row in current_details)
            total_background = sum(row["background_frames"] for row in current_details)
            summaries.append(
                {
                    "feature": feature,
                    "quality_mode": quality_mode,
                    "threshold_kind": "youden_valid_pair",
                    "threshold_directed": threshold,
                    "idle_patience_frames": idle_patience,
                    "n_passages": len(current_details),
                    "n_suited_passages": len(suited_passages),
                    "suited_passage_coverage_post": sum(
                        row["suited_passage_covered_post"] for row in suited_passages
                    )
                    / len(suited_passages),
                    "suited_passage_coverage_forward": sum(
                        row["suited_passage_covered_forward"] for row in suited_passages
                    )
                    / len(suited_passages),
                    "suited_frame_retention_post": sum(
                        row["n_suited_post_active"] for row in current_details
                    )
                    / total_suited,
                    "suited_frame_retention_forward": sum(
                        row["n_suited_forward_active"] for row in current_details
                    )
                    / total_suited,
                    "passages_activated_before_first_suited": sum(
                        row["first_activation_before_suited"] for row in suited_passages
                    ),
                    "passages_first_activation_in_partial": sum(
                        row["first_activation_in_partial"] for row in suited_passages
                    ),
                    "activation_delay_median_ms": float(np.median(delays))
                    if delays
                    else None,
                    "activation_delay_p95_ms": float(np.quantile(delays, 0.95))
                    if delays
                    else None,
                    "false_active_background_ratio": sum(
                        row["false_active_background_frames"] for row in current_details
                    )
                    / total_background,
                    "time_active_ratio_mean_by_passage": float(
                        np.mean([row["time_active_ratio"] for row in current_details])
                    ),
                    "time_active_ratio_median_by_passage": float(
                        np.median([row["time_active_ratio"] for row in current_details])
                    ),
                }
            )
    return summaries, details


def false_positive_panels(
    data_root: Path,
    pair_rows: list[dict],
    component_feature: str,
    output_dir: Path,
    limit: int = 12,
) -> list[dict]:
    selected = sorted(
        [row for row in pair_rows if row["transition"] == "background->background"],
        key=lambda row: float(row[component_feature]),
        reverse=True,
    )[:limit]
    parts = component_feature.split("__", 1)[0]
    roi_name, threshold_text = parts.rsplit("_", 1)
    threshold = float(threshold_text.removesuffix("mm"))
    fractions = ROI_DEFINITIONS[roi_name]
    panel_dir = output_dir / "background_false_positive_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for rank, row in enumerate(selected, start=1):
        previous = base.read_depth(
            data_root / "DEPTH" / row["passage_id"] / row["previous_depth_filename"]
        )
        current = base.read_depth(
            data_root / "DEPTH" / row["passage_id"] / row["depth_filename"]
        )
        diff = np.abs(current.astype(np.int32) - previous.astype(np.int32)).astype(np.float32)
        slices = base.roi_slices(diff.shape, fractions)
        roi_mask = diff[slices] >= threshold
        geometry = component_geometry(roi_mask)
        previous_image = base.depth_preview(previous)
        current_image = base.depth_preview(current)
        diff_image = base.colorize(diff)
        draw = ImageDraw.Draw(diff_image)
        y_slice, x_slice = slices
        draw.rectangle(
            (x_slice.start, y_slice.start, x_slice.stop - 1, y_slice.stop - 1),
            outline=(0, 255, 255),
            width=2,
        )
        if geometry["largest_bbox_x0"] is not None:
            draw.rectangle(
                (
                    x_slice.start + geometry["largest_bbox_x0"],
                    y_slice.start + geometry["largest_bbox_y0"],
                    x_slice.start + geometry["largest_bbox_x1"] - 1,
                    y_slice.start + geometry["largest_bbox_y1"] - 1,
                ),
                outline=(0, 255, 0),
                width=2,
            )
        header = 52
        panel = Image.new("RGB", (960, 240 + header), "white")
        for index, image in enumerate((previous_image, current_image, diff_image)):
            panel.paste(image, (index * 320, header))
        header_draw = ImageDraw.Draw(panel)
        header_draw.text(
            (6, 5),
            f"rank={rank} {row['passage_id']} #{row['capture_index']} "
            f"score={float(row[component_feature]):.4f} threshold={threshold:.0f}mm",
            fill="black",
        )
        header_draw.text(
            (6, 25),
            "previous depth | current depth | diff (cyan=ROI, green=largest component bbox)",
            fill="black",
        )
        path = panel_dir / f"rank_{rank:02d}_{row['passage_id']}_{row['capture_index']:04d}.png"
        panel.save(path)
        output.append(
            {
                "rank": rank,
                "feature": component_feature,
                "feature_value": row[component_feature],
                "passage_id": row["passage_id"],
                "capture_index": row["capture_index"],
                "delta_t_ms": row["delta_t_ms"],
                **geometry,
                "panel_path": str(path.relative_to(REPO_ROOT)),
            }
        )
    return output


def late_noise_panels(
    data_root: Path,
    indexes: dict[str, list[dict]],
    frame_rows: list[dict],
    best_noise_feature: dict,
    output_dir: Path,
    limit: int = 10,
) -> list[dict]:
    feature = best_noise_feature["feature"]
    direction = 1.0 if best_noise_feature["higher_raw_value_is_positive"] else -1.0
    threshold = best_noise_feature["recall95_threshold_directed"]
    missed = sorted(
        [
            row
            for row in frame_rows
            if row["label"] == "ruido"
            and row["elapsed_from_passage_start_ms"] > 500
            and float(row[feature]) * direction < threshold
        ],
        key=lambda row: float(row[feature]),
    )[:limit]
    panel_dir = output_dir / "late_noise_false_negative_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for rank, row in enumerate(missed, start=1):
        passage_rows = indexes[row["passage_id"]]
        index = int(row["capture_index"]) - 1
        selected = []
        selected_labels = []
        for offset in (-1, 0, 1):
            source_index = min(max(index + offset, 0), len(passage_rows) - 1)
            source = passage_rows[source_index]
            frame = base.read_depth(
                data_root / "DEPTH" / row["passage_id"] / source["depth_filename"]
            )
            selected.append(base.depth_preview(frame))
            selected_labels.append(source["label"])
        header = 52
        panel = Image.new("RGB", (960, 240 + header), "white")
        for position, image in enumerate(selected):
            panel.paste(image, (position * 320, header))
        draw = ImageDraw.Draw(panel)
        draw.text(
            (6, 5),
            f"rank={rank} {row['passage_id']} #{row['capture_index']} "
            f"elapsed={row['elapsed_from_passage_start_ms']:.0f}ms "
            f"{feature}={float(row[feature]):.3f}",
            fill="black",
        )
        draw.text(
            (6, 25),
            f"previous={selected_labels[0]} | current={selected_labels[1]} | next={selected_labels[2]}",
            fill="black",
        )
        path = panel_dir / f"rank_{rank:02d}_{row['passage_id']}_{row['capture_index']:04d}.png"
        panel.save(path)
        output.append(
            {
                "rank": rank,
                "feature": feature,
                "threshold_directed": threshold,
                **row,
                "previous_label": selected_labels[0],
                "next_label": selected_labels[2],
                "panel_path": str(path.resolve().relative_to(REPO_ROOT)),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--cohort-metrics", type=Path, default=base.DEFAULT_COHORT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort = base.load_cohort(args.cohort_metrics)
    indexes = base.load_indexes(args.data_root, cohort)

    frames = collect_frame_features(args.data_root, indexes)
    index_rows, temporal_summary, temporal_passages = label_temporal_audit(frames)
    frame_feature_names = sorted(set(frames[0]) - FRAME_METADATA)
    frame_distributions = feature_distributions(frames, frame_feature_names, "label")
    frame_performance = feature_performance(
        frames,
        frame_feature_names,
        lambda row: row["label"] == "ruido",
        lambda row: row["label"] != "ruido",
        "single_frame_noise_vs_non_noise",
    )
    best_frame_feature = frame_performance[0]
    frame_histograms = histogram_rows(
        frames,
        [row["feature"] for row in frame_performance[:8]],
    )
    histogram_images = save_histogram_images(frame_histograms, args.output_dir)
    noise_subgroups = noise_subgroup_performance(frames, best_frame_feature)
    late_noise_false_negatives = late_noise_panels(
        args.data_root,
        indexes,
        frames,
        best_frame_feature,
        args.output_dir,
    )

    pairs = collect_pair_features(args.data_root, indexes)
    pair_feature_names = sorted(
        name
        for name in set(pairs[0]) - PAIR_METADATA
        if all(
            row.get(name) is not None
            and isinstance(row.get(name), (int, float, np.integer, np.floating, bool))
            for row in pairs
        )
    )
    pair_noise_performance = feature_performance(
        pairs,
        pair_feature_names,
        lambda row: row["label"] == "ruido",
        lambda row: row["label"] != "ruido",
        "pair_current_frame_noise_vs_non_noise",
    )

    movement_performance = []
    movement_performance.extend(
        feature_performance(
            pairs,
            pair_feature_names,
            lambda row: row["label"] in {"parcial", "suited"},
            lambda row: row["label"] in {"background", "ruido"},
            "complete_current_relevant_vs_background_or_noise",
        )
    )

    key_movement_features = [
        "global__mad",
        "roi_a_y25_75_x20_80__mad",
        "roi_b_y30_70_x20_80__mad",
        "roi_a_y25_75_x20_80__changed_ratio_200mm",
        "roi_b_y30_70_x20_80__changed_ratio_200mm",
        "roi_a_y25_75_x20_80_200mm__largest_component_area_ratio",
        "roi_b_y30_70_x20_80_200mm__largest_component_area_ratio",
        "roi_b_y30_70_x20_80_200mm__largest_component_changed_fraction",
    ]
    predicted_gate_movement = movement_after_predicted_quality_gate(
        frames,
        pairs,
        frame_performance,
        key_movement_features,
    )
    valid_pairs = [
        row
        for row in pairs
        if row["previous_label"] != "ruido" and row["label"] != "ruido"
    ]
    movement_performance.extend(
        feature_performance(
            valid_pairs,
            pair_feature_names,
            lambda row: row["label"] in {"parcial", "suited"},
            lambda row: row["label"] == "background",
            "valid_pairs_current_relevant_vs_background",
        )
    )
    movement_performance.extend(
        feature_performance(
            valid_pairs,
            pair_feature_names,
            lambda row: row["previous_label"] in {"parcial", "suited"}
            or row["label"] in {"parcial", "suited"},
            lambda row: row["transition"] == "background->background",
            "valid_pairs_transition_relevant_vs_background_background",
        )
    )

    valid_current_rows = [
        row
        for row in movement_performance
        if row["evaluation"] == "valid_pairs_current_relevant_vs_background"
    ]
    base_top_features = [row["feature"] for row in valid_current_rows[:5]]
    raw_persistence_features, oracle_persistence_features = add_temporal_persistence(
        pairs, base_top_features
    )
    raw_persistence_results = persistence_performance(
        pairs,
        raw_persistence_features,
        "valid_pairs_current_relevant_with_raw_persistence",
    )
    raw_persistence_complete_results = []
    for feature in raw_persistence_features:
        finite_rows = [
            row for row in pairs if math.isfinite(float(row[feature]))
        ]
        raw_persistence_complete_results.extend(
            feature_performance(
                finite_rows,
                [feature],
                lambda row: row["label"] in {"parcial", "suited"},
                lambda row: row["label"] in {"background", "ruido"},
                "complete_current_relevant_with_raw_persistence",
            )
        )
    oracle_persistence_results = persistence_performance(
        pairs,
        oracle_persistence_features,
        "valid_pairs_current_relevant_with_oracle_quality_persistence",
    )
    movement_performance.extend(raw_persistence_results)
    movement_performance.extend(raw_persistence_complete_results)
    movement_performance.extend(oracle_persistence_results)

    candidate_rows = valid_current_rows[:3]
    candidate_rows += sorted(
        raw_persistence_results,
        key=lambda row: (-row["roc_auc"], -row["pr_auc"]),
    )[:1]
    candidate_rows += sorted(
        oracle_persistence_results,
        key=lambda row: (-row["roc_auc"], -row["pr_auc"]),
    )[:1]
    activation_summary, activation_details = passage_activation_metrics(
        indexes, pairs, candidate_rows
    )

    component_candidates = [
        row
        for row in valid_current_rows
        if "largest_component_area_ratio" in row["feature"]
        and row["feature"].startswith(("roi_a_", "roi_b_"))
    ]
    best_component = max(component_candidates, key=lambda row: row["roc_auc"])
    false_positives = false_positive_panels(
        args.data_root,
        pairs,
        best_component["feature"],
        args.output_dir,
    )

    base.write_csv(args.output_dir / "label_by_capture_index.csv", index_rows)
    base.write_csv(args.output_dir / "label_temporal_summary.csv", temporal_summary)
    base.write_csv(args.output_dir / "label_temporal_by_passage.csv", temporal_passages)
    base.write_csv(args.output_dir / "frame_features.csv", frames)
    base.write_csv(args.output_dir / "frame_feature_distributions.csv", frame_distributions)
    base.write_csv(args.output_dir / "frame_feature_histograms.csv", frame_histograms)
    base.write_csv(args.output_dir / "frame_feature_histogram_images.csv", histogram_images)
    base.write_csv(args.output_dir / "noise_single_frame_performance.csv", frame_performance)
    base.write_csv(args.output_dir / "noise_pair_performance.csv", pair_noise_performance)
    base.write_csv(args.output_dir / "noise_subgroup_performance.csv", noise_subgroups)
    base.write_csv(
        args.output_dir / "late_noise_false_negatives.csv",
        late_noise_false_negatives,
    )
    base.write_csv(args.output_dir / "pair_features.csv", pairs)
    base.write_csv(args.output_dir / "movement_signal_performance.csv", movement_performance)
    base.write_csv(
        args.output_dir / "movement_after_predicted_quality_gate.csv",
        predicted_gate_movement,
    )
    base.write_csv(args.output_dir / "movement_passage_summary.csv", activation_summary)
    base.write_csv(args.output_dir / "movement_passage_details.csv", activation_details)
    base.write_csv(args.output_dir / "background_component_false_positives.csv", false_positives)

    summary = {
        "cohort": {
            "n_passages": len(cohort),
            "n_frames": len(frames),
            "n_pairs": len(pairs),
            "label_counts": dict(Counter(row["label"] for row in frames)),
        },
        "best_single_frame_noise_feature": best_frame_feature,
        "best_pair_noise_feature": pair_noise_performance[0],
        "best_valid_movement_features": candidate_rows,
        "best_component_feature": best_component,
        "roi_definitions": ROI_DEFINITIONS,
        "pixel_thresholds_mm": PIXEL_THRESHOLDS_MM,
        "runtime_s": time.perf_counter() - started,
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
