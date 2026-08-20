#!/usr/bin/env python3
"""Auditoria diagnóstica, offline, do sinal MAD do Visual Event.

Este script não importa nem executa PADE, modelos ou o pipeline. Ele apenas lê
o cohort operacional e produz artefatos diagnósticos novos, sem sobrescrever a
calibração existente.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, stats


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domain.visual_activity import VisualActivityDetector, VisualState  # noqa: E402


DEFAULT_COHORT_METRICS = (
    REPO_ROOT
    / "power_runs/battery_mas-single_20260708_104924"
    / "mas-single_1fps_r1"
    / "mas-single_thread_2026-07-08T11:18:44.406766"
    / "metrics.json"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "data/exp1"
DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "data-analysis/visual_activity_output/visual_activity_calibration_summary.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "data-analysis/visual_event_diagnostic_output"
EXPECTED_COHORT = (184, 13_741, 1_655)
LABELS = ("background", "parcial", "ruido", "suited")
SPATIAL_TRANSITIONS = (
    "background->background",
    "background->parcial",
    "parcial->parcial",
    "parcial->suited",
    "suited->suited",
)
PIXEL_THRESHOLDS_MM = (50.0, 100.0, 200.0)
MAP_SAMPLE_SIZE = 96
RNG_SEED = 20260817


ROI_CANDIDATES = {
    "center_y20_80_x20_80": (0.20, 0.80, 0.20, 0.80),
    "center_y25_75_x25_75": (0.25, 0.75, 0.25, 0.75),
    "center_y30_70_x30_70": (0.30, 0.70, 0.30, 0.70),
    "center_y20_80_x25_75": (0.20, 0.80, 0.25, 0.75),
    "center_y25_75_x20_80": (0.25, 0.75, 0.20, 0.80),
    "center_y20_80_x30_70": (0.20, 0.80, 0.30, 0.70),
    "center_y30_70_x20_80": (0.30, 0.70, 0.20, 0.80),
}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError(f"cannot infer CSV header for empty rows: {path}")
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def quantile_summary(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def load_cohort(metrics_path: Path) -> list[str]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    animals = metrics.get("animals")
    if not isinstance(animals, dict):
        raise ValueError(f"invalid cohort metrics: {metrics_path}")
    return sorted(animals)


def load_indexes(data_root: Path, passage_ids: list[str]) -> dict[str, list[dict]]:
    indexes = {}
    for passage_id in passage_ids:
        path = data_root / "animal-tags" / passage_id / "simulation_index.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows.sort(key=lambda row: float(row["relative_time_ms"]))
        indexes[passage_id] = rows
    return indexes


def load_thresholds(calibration_path: Path) -> list[tuple[float, int]]:
    configurations = set()
    with calibration_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["configuration"] != "mad":
                continue
            configurations.add(
                (float(row["threshold"]), int(row["idle_patience_frames"]))
            )
    return sorted(configurations)


def read_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint16)


def transition_name(previous_label: str, current_label: str) -> str:
    return f"{previous_label}->{current_label}"


def roi_slices(shape: tuple[int, int], fractions: tuple[float, float, float, float]):
    height, width = shape
    y0, y1, x0, x1 = fractions
    return (
        slice(int(round(y0 * height)), int(round(y1 * height))),
        slice(int(round(x0 * width)), int(round(x1 * width))),
    )


def region_definitions(shape: tuple[int, int]) -> dict[str, tuple[slice, slice]]:
    height, width = shape
    regions = {}
    for index in range(5):
        regions[f"horizontal_y{index * 20}_{(index + 1) * 20}"] = (
            slice(index * height // 5, (index + 1) * height // 5),
            slice(0, width),
        )
        regions[f"vertical_x{index * 20}_{(index + 1) * 20}"] = (
            slice(0, height),
            slice(index * width // 5, (index + 1) * width // 5),
        )
    for low, high in ((20, 80), (25, 75), (30, 70)):
        regions[f"center_y{low}_{high}_x{low}_{high}"] = roi_slices(
            shape, (low / 100, high / 100, low / 100, high / 100)
        )
    return regions


def reservoir_add(
    reservoir: list[np.ndarray],
    item: np.ndarray,
    seen: int,
    limit: int,
    rng: np.random.Generator,
) -> None:
    if len(reservoir) < limit:
        reservoir.append(item.copy())
        return
    replacement = int(rng.integers(0, seen))
    if replacement < limit:
        reservoir[replacement] = item.copy()


def collect_pairs(
    data_root: Path,
    indexes: dict[str, list[dict]],
) -> tuple[list[dict], dict, dict, dict, tuple[int, int], dict]:
    records: list[dict] = []
    spatial_sums: dict[str, np.ndarray] = {}
    spatial_counts = defaultdict(int)
    spatial_samples = defaultdict(list)
    region_values = defaultdict(list)
    roi_values = defaultdict(list)
    rng = np.random.default_rng(RNG_SEED)
    shape = None
    depth_min = math.inf
    depth_max = -math.inf
    zero_pixels = 0
    depth_pixels = 0

    for passage_id, rows in indexes.items():
        previous = None
        previous_row = None
        for capture_index, row in enumerate(rows, start=1):
            path = data_root / "DEPTH" / passage_id / row["depth_filename"]
            current = read_depth(path)
            if shape is None:
                shape = current.shape
            elif current.shape != shape:
                raise ValueError(f"inconsistent depth shape: {path} {current.shape} != {shape}")
            depth_min = min(depth_min, int(current.min()))
            depth_max = max(depth_max, int(current.max()))
            zero_pixels += int(np.count_nonzero(current == 0))
            depth_pixels += int(current.size)
            if previous is not None:
                diff = np.abs(
                    current.astype(np.int32, copy=False)
                    - previous.astype(np.int32, copy=False)
                ).astype(np.float32)
                transition = transition_name(previous_row["label"], row["label"])
                total_energy = float(diff.sum(dtype=np.float64))
                record = {
                    "passage_id": passage_id,
                    "capture_index": capture_index,
                    "previous_timestamp_ms": float(previous_row["relative_time_ms"]),
                    "timestamp_ms": float(row["relative_time_ms"]),
                    "delta_t_ms": float(row["relative_time_ms"])
                    - float(previous_row["relative_time_ms"]),
                    "previous_label": previous_row["label"],
                    "label": row["label"],
                    "transition": transition,
                    "previous_depth_filename": previous_row["depth_filename"],
                    "depth_filename": row["depth_filename"],
                    # Preserve exatamente a redução float32 usada pelo detector
                    # operacional; float64 aqui pode mudar pares na fronteira.
                    "mad": float(np.mean(diff)),
                }

                for name, slices in region_definitions(diff.shape).items():
                    value = float(diff[slices].mean(dtype=np.float64))
                    region_values[(transition, name)].append(value)

                for name, fractions in ROI_CANDIDATES.items():
                    slices = roi_slices(diff.shape, fractions)
                    region = diff[slices]
                    roi_mad = float(region.mean(dtype=np.float64))
                    energy_fraction = (
                        float(region.sum(dtype=np.float64)) / total_energy
                        if total_energy > 0
                        else 0.0
                    )
                    roi_values[(transition, name, "mad")].append(roi_mad)
                    roi_values[(transition, name, "energy_fraction")].append(
                        energy_fraction
                    )

                if transition in SPATIAL_TRANSITIONS:
                    if transition not in spatial_sums:
                        spatial_sums[transition] = np.zeros(diff.shape, dtype=np.float64)
                    spatial_sums[transition] += diff
                    spatial_counts[transition] += 1
                    reservoir_add(
                        spatial_samples[transition],
                        diff.astype(np.uint16),
                        spatial_counts[transition],
                        MAP_SAMPLE_SIZE,
                        rng,
                    )
                records.append(record)
            previous = current
            previous_row = row

    if shape is None:
        raise ValueError("empty cohort")
    depth_audit = {
        "shape_height": shape[0],
        "shape_width": shape[1],
        "dtype": "uint16",
        "minimum_mm": depth_min,
        "maximum_mm": depth_max,
        "zero_pixels": zero_pixels,
        "total_pixels": depth_pixels,
        "zero_pixel_ratio": zero_pixels / depth_pixels,
    }
    spatial = {
        "sums": spatial_sums,
        "counts": dict(spatial_counts),
        "samples": dict(spatial_samples),
    }
    return records, spatial, region_values, roi_values, shape, depth_audit


def audit_rgb(data_root: Path, indexes: dict[str, list[dict]]) -> dict:
    total = filenames = present = aligned_stems = 0
    shapes = defaultdict(int)
    modes = defaultdict(int)
    missing_examples = []
    for passage_id, rows in indexes.items():
        for row in rows:
            total += 1
            rgb_filename = row.get("rgb_filename")
            filenames += bool(rgb_filename)
            if rgb_filename:
                depth_stem = row["depth_filename"].replace("_DEPTH_320_240_1.png", "")
                rgb_stem = rgb_filename.replace("_RGB_640_480_3.png", "")
                aligned_stems += depth_stem == rgb_stem
                path = data_root / "RGB" / passage_id / rgb_filename
                if path.is_file():
                    present += 1
                    with Image.open(path) as image:
                        shapes[str(image.size)] += 1
                        modes[image.mode] += 1
                elif len(missing_examples) < 5:
                    missing_examples.append(str(path.relative_to(REPO_ROOT)))
    return {
        "n_index_entries": total,
        "n_rgb_filenames": filenames,
        "n_rgb_files_present": present,
        "n_rgb_files_missing": total - present,
        "n_depth_rgb_aligned_stems": aligned_stems,
        "shared_index_timestamp": True,
        "observed_shapes": dict(shapes),
        "observed_modes": dict(modes),
        "filename_declared_shape": "480x640x3",
        "filename_declared_format": "PNG",
        "actual_dtype": None if not present else "see observed_modes",
        "missing_examples": missing_examples,
    }


def transition_distribution(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[record["transition"]].append(record["mad"])
    rows = []
    for previous in LABELS:
        for current in LABELS:
            transition = transition_name(previous, current)
            summary = quantile_summary(groups[transition])
            rows.append(
                {
                    "previous_label": previous,
                    "label": current,
                    "transition": transition,
                    **summary,
                }
            )
    return rows


def delta_t_audit(records: list[dict]) -> list[dict]:
    rows = []
    groups = {"global": records}
    for transition in sorted({row["transition"] for row in records}):
        groups[transition] = [row for row in records if row["transition"] == transition]
    for name, values in groups.items():
        delta = np.asarray([row["delta_t_ms"] for row in values], dtype=np.float64)
        mad = np.asarray([row["mad"] for row in values], dtype=np.float64)
        if len(values) >= 3 and np.ptp(delta) > 0 and np.ptp(mad) > 0:
            pearson = stats.pearsonr(delta, mad)
            spearman = stats.spearmanr(delta, mad)
            pearson_r, pearson_p = float(pearson.statistic), float(pearson.pvalue)
            spearman_r, spearman_p = float(spearman.statistic), float(spearman.pvalue)
        else:
            pearson_r = pearson_p = spearman_r = spearman_p = None
        rows.append(
            {
                "group": name,
                "n_pairs": len(values),
                "delta_t_median_ms": float(np.median(delta)) if delta.size else None,
                "delta_t_p95_ms": float(np.quantile(delta, 0.95)) if delta.size else None,
                "delta_t_p99_ms": float(np.quantile(delta, 0.99)) if delta.size else None,
                "delta_t_max_ms": float(np.max(delta)) if delta.size else None,
                "mad_median": float(np.median(mad)) if mad.size else None,
                "mad_p95": float(np.quantile(mad, 0.95)) if mad.size else None,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )
    return rows


def evaluate_active_ratios(
    indexes: dict[str, list[dict]],
    pairs: list[dict],
    configurations: list[tuple[float, int]],
) -> tuple[list[dict], list[dict]]:
    pair_by_passage = defaultdict(list)
    for pair in pairs:
        pair_by_passage[pair["passage_id"]].append(pair)
    passage_rows = []
    summary_rows = []
    for threshold, patience in configurations:
        for passage_id, frames in indexes.items():
            detector = VisualActivityDetector(threshold, patience)
            states = [VisualState.IDLE]
            passage_pairs = sorted(pair_by_passage[passage_id], key=lambda row: row["capture_index"])
            states.extend(detector.observe_mad(row["mad"]).visual_state for row in passage_pairs)
            if len(states) != len(frames):
                raise AssertionError(f"state/frame mismatch for {passage_id}")
            timestamps = np.asarray([float(row["relative_time_ms"]) for row in frames])
            intervals = np.diff(timestamps)
            active_flags = np.asarray([state is VisualState.ACTIVE for state in states])
            active_time_ms = float(intervals[active_flags[:-1]].sum())
            total_time_ms = float(intervals.sum())
            passage_rows.append(
                {
                    "threshold": threshold,
                    "idle_patience_frames": patience,
                    "passage_id": passage_id,
                    "n_frames": len(frames),
                    "active_frames": int(active_flags.sum()),
                    "frame_active_ratio": float(active_flags.mean()),
                    "duration_ms": total_time_ms,
                    "active_time_ms": active_time_ms,
                    "time_active_ratio": active_time_ms / total_time_ms if total_time_ms else 0.0,
                    "frame_minus_time_ratio": float(active_flags.mean())
                    - (active_time_ms / total_time_ms if total_time_ms else 0.0),
                }
            )

        current = [
            row
            for row in passage_rows
            if row["threshold"] == threshold
            and row["idle_patience_frames"] == patience
        ]
        frame_ratios = [row["frame_active_ratio"] for row in current]
        time_ratios = [row["time_active_ratio"] for row in current]
        total_frames = sum(row["n_frames"] for row in current)
        total_active_frames = sum(row["active_frames"] for row in current)
        total_time = sum(row["duration_ms"] for row in current)
        total_active_time = sum(row["active_time_ms"] for row in current)
        frame_summary = quantile_summary(frame_ratios)
        time_summary = quantile_summary(time_ratios)
        summary_rows.append(
            {
                "threshold": threshold,
                "idle_patience_frames": patience,
                "n_passages": len(current),
                "frame_active_ratio_global": total_active_frames / total_frames,
                "time_active_ratio_global": total_active_time / total_time,
                "global_frame_minus_time_ratio": total_active_frames / total_frames
                - total_active_time / total_time,
                "frame_ratio_p25_by_passage": frame_summary["p25"],
                "frame_ratio_median_by_passage": frame_summary["median"],
                "frame_ratio_p75_by_passage": frame_summary["p75"],
                "frame_ratio_p95_by_passage": frame_summary["p95"],
                "time_ratio_p25_by_passage": time_summary["p25"],
                "time_ratio_median_by_passage": time_summary["median"],
                "time_ratio_p75_by_passage": time_summary["p75"],
                "time_ratio_p95_by_passage": time_summary["p95"],
            }
        )
    return summary_rows, passage_rows


def largest_active_ratio_differences(passage_rows: list[dict], limit: int = 10) -> list[dict]:
    output = []
    configurations = sorted(
        {
            (row["threshold"], row["idle_patience_frames"])
            for row in passage_rows
        }
    )
    for threshold, patience in configurations:
        selected = [
            row
            for row in passage_rows
            if row["threshold"] == threshold
            and row["idle_patience_frames"] == patience
        ]
        selected.sort(key=lambda row: abs(row["frame_minus_time_ratio"]), reverse=True)
        for rank, row in enumerate(selected[:limit], start=1):
            output.append({"difference_rank": rank, **row})
    return output


def spatial_region_summary(region_values: dict) -> list[dict]:
    rows = []
    for (transition, region), values in sorted(region_values.items()):
        rows.append({"transition": transition, "region": region, **quantile_summary(values)})
    return rows


def roi_summary(roi_values: dict, shape: tuple[int, int]) -> list[dict]:
    rows = []
    transitions = sorted({key[0] for key in roi_values})
    for name, fractions in ROI_CANDIDATES.items():
        target_mad = []
        target_energy = []
        background_mad = []
        background_energy = []
        noise_mad = []
        noise_energy = []
        for transition in transitions:
            previous, current = transition.split("->")
            mad = roi_values[(transition, name, "mad")]
            energy = roi_values[(transition, name, "energy_fraction")]
            if previous in {"parcial", "suited"} or current in {"parcial", "suited"}:
                target_mad.extend(mad)
                target_energy.extend(energy)
            if transition == "background->background":
                background_mad.extend(mad)
                background_energy.extend(energy)
            if previous == "ruido" or current == "ruido":
                noise_mad.extend(mad)
                noise_energy.extend(energy)
        y_slice, x_slice = roi_slices(shape, fractions)
        area = (y_slice.stop - y_slice.start) * (x_slice.stop - x_slice.start)
        area_fraction = area / (shape[0] * shape[1])
        target_median = float(np.median(target_mad))
        background_median = float(np.median(background_mad))
        rows.append(
            {
                "roi": name,
                "y0_fraction": fractions[0],
                "y1_fraction": fractions[1],
                "x0_fraction": fractions[2],
                "x1_fraction": fractions[3],
                "area_fraction": area_fraction,
                "target_mad_median": target_median,
                "background_background_mad_median": background_median,
                "noise_associated_mad_median": float(np.median(noise_mad)),
                "target_to_background_median_ratio": target_median / background_median
                if background_median
                else None,
                "target_energy_fraction_median": float(np.median(target_energy)),
                "background_energy_fraction_median": float(np.median(background_energy)),
                "noise_energy_fraction_median": float(np.median(noise_energy)),
                "target_energy_density_gain": float(np.median(target_energy)) / area_fraction,
                "background_energy_density_gain": float(np.median(background_energy))
                / area_fraction,
            }
        )
    return rows


def choose_exploratory_roi(rows: list[dict]) -> dict:
    # Apenas escolhe a ROI usada para comparar sinais no diagnóstico. Não congela
    # configuração do detector e todas as candidatas permanecem no CSV.
    return max(
        rows,
        key=lambda row: (
            row["target_to_background_median_ratio"]
            * row["target_energy_fraction_median"]
        ),
    )


def largest_component(mask: np.ndarray) -> dict:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return {
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_fraction_of_frame": 0.0,
            "largest_component_fraction_of_changed": 0.0,
            "bbox_x0": None,
            "bbox_y0": None,
            "bbox_x1": None,
            "bbox_y1": None,
            "centroid_x": None,
            "centroid_y": None,
        }
    sizes = np.bincount(labels.ravel())[1:]
    largest_label = int(np.argmax(sizes)) + 1
    largest_size = int(sizes[largest_label - 1])
    ys, xs = np.nonzero(labels == largest_label)
    changed = int(np.count_nonzero(mask))
    return {
        "component_count": int(count),
        "largest_component_pixels": largest_size,
        "largest_component_fraction_of_frame": largest_size / mask.size,
        "largest_component_fraction_of_changed": largest_size / changed if changed else 0.0,
        "bbox_x0": int(xs.min()),
        "bbox_y0": int(ys.min()),
        "bbox_x1": int(xs.max()) + 1,
        "bbox_y1": int(ys.max()) + 1,
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
    }


def movement_metrics(
    data_root: Path,
    indexes: dict[str, list[dict]],
    roi: dict,
) -> list[dict]:
    output = []
    fractions = (
        roi["y0_fraction"],
        roi["y1_fraction"],
        roi["x0_fraction"],
        roi["x1_fraction"],
    )
    for passage_id, rows in indexes.items():
        previous = None
        previous_row = None
        for capture_index, row in enumerate(rows, start=1):
            current = read_depth(data_root / "DEPTH" / passage_id / row["depth_filename"])
            if previous is not None:
                diff = np.abs(
                    current.astype(np.int32, copy=False)
                    - previous.astype(np.int32, copy=False)
                ).astype(np.float32)
                roi_slice = roi_slices(diff.shape, fractions)
                for threshold in PIXEL_THRESHOLDS_MM:
                    mask = diff >= threshold
                    roi_mask = mask[roi_slice]
                    component = largest_component(mask)
                    roi_component = largest_component(roi_mask)
                    output.append(
                        {
                            "passage_id": passage_id,
                            "capture_index": capture_index,
                            "previous_label": previous_row["label"],
                            "label": row["label"],
                            "transition": transition_name(previous_row["label"], row["label"]),
                            "delta_t_ms": float(row["relative_time_ms"])
                            - float(previous_row["relative_time_ms"]),
                            "pixel_threshold_mm": threshold,
                            "global_mad": float(diff.mean(dtype=np.float64)),
                            "roi": roi["roi"],
                            "roi_mad": float(diff[roi_slice].mean(dtype=np.float64)),
                            "changed_pixel_ratio": float(mask.mean()),
                            "changed_pixel_ratio_roi": float(roi_mask.mean()),
                            **component,
                            "roi_component_count": roi_component["component_count"],
                            "roi_largest_component_pixels": roi_component[
                                "largest_component_pixels"
                            ],
                            "roi_largest_component_fraction": roi_component[
                                "largest_component_fraction_of_frame"
                            ],
                            "roi_largest_component_fraction_of_changed": roi_component[
                                "largest_component_fraction_of_changed"
                            ],
                        }
                    )
            previous = current
            previous_row = row
    return output


def rank_auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    values = np.asarray(positive + negative, dtype=np.float64)
    ranks = stats.rankdata(values)
    n_pos = len(positive)
    n_neg = len(negative)
    rank_sum = float(ranks[:n_pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def signal_comparison(movement: list[dict]) -> list[dict]:
    rows = []
    signals = (
        "global_mad",
        "roi_mad",
        "changed_pixel_ratio",
        "changed_pixel_ratio_roi",
        "largest_component_fraction_of_frame",
        "roi_largest_component_fraction",
    )
    for threshold in PIXEL_THRESHOLDS_MM:
        selected = [row for row in movement if row["pixel_threshold_mm"] == threshold]
        groups = {
            "relevant_partial_or_suited": [
                row
                for row in selected
                if row["previous_label"] in {"parcial", "suited"}
                or row["label"] in {"parcial", "suited"}
            ],
            "background_background": [
                row for row in selected if row["transition"] == "background->background"
            ],
            "noise_associated": [
                row
                for row in selected
                if row["previous_label"] == "ruido" or row["label"] == "ruido"
            ],
        }
        for signal in signals:
            positive = [row[signal] for row in groups["relevant_partial_or_suited"]]
            negative = [row[signal] for row in groups["background_background"]]
            auc = rank_auc(positive, negative)
            for group, group_rows in groups.items():
                rows.append(
                    {
                        "pixel_threshold_mm": threshold,
                        "signal": signal,
                        "group": group,
                        "auc_relevant_vs_background_background": auc,
                        **quantile_summary(row[signal] for row in group_rows),
                    }
                )
    return rows


def colorize(values: np.ndarray, vmax: float | None = None) -> Image.Image:
    vmax = float(vmax if vmax is not None else np.quantile(values, 0.99))
    if vmax <= 0:
        vmax = 1.0
    normalized = np.clip(values / vmax, 0.0, 1.0)
    stops = np.asarray(
        [
            [0, 0, 4],
            [59, 15, 112],
            [140, 41, 129],
            [221, 73, 104],
            [253, 159, 108],
            [252, 253, 191],
        ],
        dtype=np.float32,
    )
    scaled = normalized * (len(stops) - 1)
    low = np.floor(scaled).astype(int)
    high = np.minimum(low + 1, len(stops) - 1)
    alpha = (scaled - low)[..., None]
    rgb = stops[low] * (1.0 - alpha) + stops[high] * alpha
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def depth_preview(depth: np.ndarray) -> Image.Image:
    normalized = 1.0 - np.clip((depth.astype(np.float32) - 10.0) / 2590.0, 0.0, 1.0)
    gray = (normalized * 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def save_spatial_maps(spatial: dict, output_dir: Path) -> list[dict]:
    map_dir = output_dir / "heatmaps"
    map_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    metadata = []
    for transition in SPATIAL_TRANSITIONS:
        count = spatial["counts"].get(transition, 0)
        if not count:
            continue
        sample = np.stack(spatial["samples"][transition]).astype(np.float32)
        maps = {
            "mean": (spatial["sums"][transition] / count).astype(np.float32),
            "median_sampled": np.median(sample, axis=0).astype(np.float32),
            "p90_sampled": np.quantile(sample, 0.90, axis=0).astype(np.float32),
        }
        safe_transition = transition.replace("->", "_to_")
        for statistic, values in maps.items():
            key = f"{safe_transition}_{statistic}"
            arrays[key] = values
            vmax = float(np.quantile(values, 0.99))
            path = map_dir / f"{key}.png"
            colorize(values, vmax).save(path)
            metadata.append(
                {
                    "transition": transition,
                    "statistic": statistic,
                    "n_pairs": count,
                    "sample_size_for_quantile_map": len(sample)
                    if "sampled" in statistic
                    else None,
                    "color_scale_vmax_p99": vmax,
                    "path": str(path.relative_to(REPO_ROOT)),
                }
            )
    np.savez_compressed(output_dir / "spatial_maps.npz", **arrays)
    return metadata


def representative_records(records: list[dict]) -> list[tuple[str, str, dict]]:
    selections = []
    definitions = {
        "background_background": [
            row for row in records if row["transition"] == "background->background"
        ],
        "noise_associated": [
            row
            for row in records
            if row["previous_label"] == "ruido" or row["label"] == "ruido"
        ],
    }
    quantiles = (
        ("minimum", 0.0),
        ("median", 0.5),
        ("p90", 0.90),
        ("p95", 0.95),
        ("p99", 0.99),
        ("maximum", 1.0),
    )
    for group, rows in definitions.items():
        ordered = sorted(rows, key=lambda row: row["mad"])
        values = np.asarray([row["mad"] for row in ordered])
        for name, quantile in quantiles:
            target = float(np.quantile(values, quantile))
            selected = min(ordered, key=lambda row: abs(row["mad"] - target))
            selections.append((group, name, selected))
    return selections


def representative_panels(
    data_root: Path,
    selections: list[tuple[str, str, dict]],
    roi: dict,
    output_dir: Path,
) -> list[dict]:
    panel_dir = output_dir / "representative_pairs"
    panel_dir.mkdir(parents=True, exist_ok=True)
    output = []
    fractions = (
        roi["y0_fraction"],
        roi["y1_fraction"],
        roi["x0_fraction"],
        roi["x1_fraction"],
    )
    for group, quantile, row in selections:
        passage = row["passage_id"]
        previous = read_depth(data_root / "DEPTH" / passage / row["previous_depth_filename"])
        current = read_depth(data_root / "DEPTH" / passage / row["depth_filename"])
        diff = np.abs(current.astype(np.int32) - previous.astype(np.int32)).astype(np.float32)
        mask = diff >= 100.0
        component = largest_component(mask)
        roi_slice = roi_slices(diff.shape, fractions)
        center_energy = float(diff[roi_slice].sum(dtype=np.float64))
        total_energy = float(diff.sum(dtype=np.float64))

        images = [depth_preview(previous), depth_preview(current), colorize(diff)]
        width = sum(image.width for image in images)
        header = 52
        panel = Image.new("RGB", (width, images[0].height + header), "white")
        x = 0
        for image in images:
            panel.paste(image, (x, header))
            x += image.width
        draw = ImageDraw.Draw(panel)
        title = (
            f"{group} {quantile} | {passage} #{row['capture_index']} | "
            f"{row['transition']} | MAD={row['mad']:.2f} mm | dt={row['delta_t_ms']:.0f} ms"
        )
        draw.text((6, 5), title, fill="black")
        draw.text((6, 25), "previous depth", fill="black")
        draw.text((326, 25), "current depth", fill="black")
        draw.text((646, 25), "absolute difference", fill="black")
        path = panel_dir / f"{group}_{quantile}_{passage}_{row['capture_index']:04d}.png"
        panel.save(path)
        output.append(
            {
                "group": group,
                "quantile": quantile,
                **row,
                "roi": roi["roi"],
                "roi_energy_fraction": center_energy / total_energy if total_energy else 0.0,
                "changed_pixel_ratio_100mm": float(mask.mean()),
                **component,
                "panel_path": str(path.relative_to(REPO_ROOT)),
            }
        )
    return output


def cohort_audit(
    data_root: Path,
    cohort: list[str],
    indexes: dict[str, list[dict]],
    depth_audit: dict,
    rgb_audit: dict,
) -> dict:
    total_frames = sum(len(rows) for rows in indexes.values())
    suited = sum(row["label"] == "suited" for rows in indexes.values() for row in rows)
    observed = (len(cohort), total_frames, suited)
    if observed != EXPECTED_COHORT:
        raise ValueError(f"cohort mismatch: {observed} != {EXPECTED_COHORT}")
    labels = {
        label: sum(row["label"] == label for rows in indexes.values() for row in rows)
        for label in LABELS
    }
    all_passages = sorted(
        path.parent.name
        for path in (data_root / "animal-tags").glob("*/simulation_index.json")
    )
    excluded = sorted(set(all_passages) - set(cohort))
    return {
        "n_passages_full_dataset": len(all_passages),
        "n_passages": len(cohort),
        "excluded_passage_ids": excluded,
        "n_frames": total_frames,
        "n_suited_frames": suited,
        "label_counts": labels,
        "depth": depth_audit,
        "rgb": rgb_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cohort-metrics", type=Path, default=DEFAULT_COHORT_METRICS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(args.cohort_metrics)
    indexes = load_indexes(args.data_root, cohort)
    rgb = audit_rgb(args.data_root, indexes)
    pairs, spatial, region_values, roi_values, shape, depth = collect_pairs(
        args.data_root, indexes
    )
    audit = cohort_audit(args.data_root, cohort, indexes, depth, rgb)

    transitions = transition_distribution(pairs)
    delta = delta_t_audit(pairs)
    configurations = load_thresholds(args.calibration)
    active_summary, active_by_passage = evaluate_active_ratios(
        indexes, pairs, configurations
    )
    largest_active_differences = largest_active_ratio_differences(active_by_passage)
    spatial_regions = spatial_region_summary(region_values)
    rois = roi_summary(roi_values, shape)
    selected_roi = choose_exploratory_roi(rois)
    movement = movement_metrics(args.data_root, indexes, selected_roi)
    signals = signal_comparison(movement)
    map_metadata = save_spatial_maps(spatial, args.output_dir)
    representatives = representative_panels(
        args.data_root,
        representative_records(pairs),
        selected_roi,
        args.output_dir,
    )

    write_csv(args.output_dir / "active_ratio_audit.csv", active_summary)
    write_csv(args.output_dir / "active_ratio_by_passage.csv", active_by_passage)
    write_csv(
        args.output_dir / "active_ratio_largest_differences.csv",
        largest_active_differences,
    )
    write_csv(args.output_dir / "mad_transition_distribution.csv", transitions)
    write_csv(args.output_dir / "mad_delta_t_audit.csv", delta)
    write_csv(args.output_dir / "mad_spatial_regions.csv", spatial_regions)
    write_csv(args.output_dir / "roi_candidate_summary.csv", rois)
    write_csv(args.output_dir / "movement_region_metrics.csv", movement)
    write_csv(args.output_dir / "signal_comparison.csv", signals)
    write_csv(args.output_dir / "spatial_map_metadata.csv", map_metadata)
    write_csv(args.output_dir / "representative_pairs.csv", representatives)
    write_csv(
        args.output_dir / "rgb_dataset_audit.csv",
        [
            {
                **{key: value for key, value in rgb.items() if key != "missing_examples"},
                "missing_examples": ";".join(rgb["missing_examples"]),
            }
        ],
    )
    elapsed = time.perf_counter() - started
    summary = {
        "cohort": audit,
        "pair_count": len(pairs),
        "active_ratio_definition": "active frames / all frames; state after observing each frame",
        "time_ratio_interval_convention": (
            "state after observing F(t) owns [timestamp(F(t)), timestamp(F(t+1))); "
            "last frame owns no unobserved tail"
        ),
        "exploratory_roi_for_signal_comparison": selected_roi,
        "spatial_quantile_maps_are_sampled": True,
        "spatial_quantile_map_sample_size_max": MAP_SAMPLE_SIZE,
        "pixel_thresholds_mm": list(PIXEL_THRESHOLDS_MM),
        "runtime_s": elapsed,
    }
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
