#!/usr/bin/env python3
"""Auditoria offline dos labels para definir o target do Visual Event.

O script apenas le o cohort operacional e artefatos das auditorias PDI
anteriores. Nao executa modelos, PADE ou o pipeline e grava tudo em um
diretorio de analise isolado.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import visual_event_diagnostic as base  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "data-analysis/visual_event_label_audit_output"
DEFAULT_PAIR_FEATURES = (
    REPO_ROOT / "data-analysis/visual_event_noise_pdi_output/pair_features.csv"
)
EXPECTED_COHORT = (184, 13_741, 1_655)
RELEVANT_LABELS = {"parcial", "suited"}
VALID_LABELS = {"background", "parcial", "suited"}
BEST_PDI_FEATURE = (
    "roi_b_y30_70_x20_80_200mm__largest_component_changed_fraction"
)
BEST_PDI_EXPLORATORY_THRESHOLD = 0.08747855917667238
ROI_B = (0.30, 0.70, 0.20, 0.80)

# Preenchido somente depois da inspecao humana dos paineis. As categorias nao
# alteram os labels originais e sao deliberadamente mantidas na camada offline.
MANUAL_CLASSIFICATIONS: dict[str, tuple[str, str, str]] = {
    f"S{index:03d}": (
        "clearly_empty",
        "not_applicable",
        "sem parte de animal claramente discernivel no depth",
    )
    for index in range(1, 101)
}
MANUAL_CLASSIFICATIONS.update(
    {
        "S010": ("ambiguous_possible_animal", "sufficient", "massa difusa proxima da entrada"),
        "S020": ("ambiguous_possible_animal", "sufficient", "alteracao vertical difusa no centro"),
        "S030": ("ambiguous_possible_animal", "sufficient", "objeto claro pontual; animal ou artefato incerto"),
        "S062": ("ambiguous_possible_animal", "partial", "estrutura vertical clara parcialmente dentro da ROI"),
        "S068": ("ambiguous_possible_animal", "partial", "massa clara discreta proxima da borda"),
        "S080": ("clearly_animal_visible", "partial", "corpo/pernas ainda visiveis apos ultimo parcial"),
        "S082": ("clearly_animal_visible", "partial", "massa corporal visivel antes do primeiro parcial"),
        "S083": ("clearly_animal_visible", "sufficient", "pernas/corpo ainda visiveis apos ultimo suited"),
        "S084": ("clearly_animal_visible", "partial", "massa corporal na borda inferior antes do parcial"),
        "S085": ("clearly_animal_visible", "partial", "massa corporal na entrada antes do parcial"),
        "S088": ("ambiguous_possible_animal", "partial", "regiao clara inferior; presenca possivel"),
        "S089": ("clearly_animal_visible", "partial", "massa corporal difusa antes do parcial"),
        "S091": ("clearly_animal_visible", "sufficient", "animal claramente ocupa a ROI antes do primeiro suited"),
        "S093": ("ambiguous_possible_animal", "partial", "massa corporal difusa antes do parcial"),
        "S094": ("clearly_animal_visible", "partial", "massa corporal visivel antes do parcial"),
        "S096": ("clearly_animal_visible", "partial", "massa corporal visivel na entrada"),
        "S097": ("ambiguous_possible_animal", "sufficient", "regiao difusa central compativel com entrada"),
        "S100": ("ambiguous_possible_animal", "sufficient", "objeto claro pequeno; animal ou artefato incerto"),
    }
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def collapse_labels(rows: list[dict], include_noise: bool) -> str:
    labels = [
        row["label"]
        for row in rows
        if include_noise or row["label"] != "ruido"
    ]
    collapsed: list[str] = []
    for label in labels:
        if not collapsed or collapsed[-1] != label:
            collapsed.append(label)
    return "->".join(collapsed)


def recording_date(depth_filename: str) -> str | None:
    match = re.search(r"_(\d{4})_(\d{2})_(\d{2})_", depth_filename)
    return None if match is None else "-".join(match.groups())


def build_frame_rows(indexes: dict[str, list[dict]]) -> list[dict]:
    output = []
    for passage_id, rows in indexes.items():
        start = float(rows[0]["relative_time_ms"])
        for position, row in enumerate(rows):
            timestamp = float(row["relative_time_ms"])
            next_timestamp = (
                float(rows[position + 1]["relative_time_ms"])
                if position + 1 < len(rows)
                else timestamp
            )
            output.append(
                {
                    "passage_id": passage_id,
                    "capture_index": position + 1,
                    "zero_based_index": position,
                    "relative_time_ms": timestamp,
                    "elapsed_from_passage_start_ms": timestamp - start,
                    "interval_to_next_ms": max(0.0, next_timestamp - timestamp),
                    "label": row["label"],
                    "depth_filename": row["depth_filename"],
                    "rgb_filename": row.get("rgb_filename"),
                    "recording_date": recording_date(row["depth_filename"]),
                }
            )
    return output


def landmark_index(rows: list[dict], label: str, first: bool) -> int | None:
    matches = [index for index, row in enumerate(rows) if row["label"] == label]
    return (matches[0] if first else matches[-1]) if matches else None


def passage_landmarks(indexes: dict[str, list[dict]]) -> list[dict]:
    output = []
    for passage_id, rows in indexes.items():
        first_valid = next(
            (i for i, row in enumerate(rows) if row["label"] != "ruido"), None
        )
        first_partial = landmark_index(rows, "parcial", True)
        first_suited = landmark_index(rows, "suited", True)
        last_suited = landmark_index(rows, "suited", False)
        last_partial = landmark_index(rows, "parcial", False)
        first_return_background = (
            next(
                (
                    i
                    for i in range((last_partial or 0) + 1, len(rows))
                    if rows[i]["label"] == "background"
                ),
                None,
            )
            if last_partial is not None
            else None
        )

        def value(index: int | None, field: str):
            return None if index is None else rows[index][field]

        output.append(
            {
                "passage_id": passage_id,
                "n_frames": len(rows),
                "recording_date": recording_date(rows[0]["depth_filename"]),
                "sequence_with_noise": collapse_labels(rows, True),
                "sequence_valid_only": collapse_labels(rows, False),
                "first_valid_capture_index": None if first_valid is None else first_valid + 1,
                "first_valid_timestamp_ms": value(first_valid, "relative_time_ms"),
                "first_partial_capture_index": None if first_partial is None else first_partial + 1,
                "first_partial_timestamp_ms": value(first_partial, "relative_time_ms"),
                "first_suited_capture_index": None if first_suited is None else first_suited + 1,
                "first_suited_timestamp_ms": value(first_suited, "relative_time_ms"),
                "last_suited_capture_index": None if last_suited is None else last_suited + 1,
                "last_suited_timestamp_ms": value(last_suited, "relative_time_ms"),
                "last_partial_capture_index": None if last_partial is None else last_partial + 1,
                "last_partial_timestamp_ms": value(last_partial, "relative_time_ms"),
                "first_return_background_capture_index": (
                    None if first_return_background is None else first_return_background + 1
                ),
                "first_return_background_timestamp_ms": value(
                    first_return_background, "relative_time_ms"
                ),
                "first_partial_to_first_suited_frames": (
                    None
                    if first_partial is None or first_suited is None
                    else first_suited - first_partial
                ),
                "first_partial_to_first_suited_ms": (
                    None
                    if first_partial is None or first_suited is None
                    else float(rows[first_suited]["relative_time_ms"])
                    - float(rows[first_partial]["relative_time_ms"])
                ),
                "last_suited_to_last_partial_frames": (
                    None
                    if last_suited is None or last_partial is None
                    else last_partial - last_suited
                ),
                "last_suited_to_last_partial_ms": (
                    None
                    if last_suited is None or last_partial is None
                    else float(rows[last_partial]["relative_time_ms"])
                    - float(rows[last_suited]["relative_time_ms"])
                ),
            }
        )
    return output


def pattern_summary(landmarks: list[dict], field: str) -> list[dict]:
    counter = Counter(row[field] for row in landmarks)
    total = len(landmarks)
    return [
        {
            "sequence_scope": field,
            "rank": rank,
            "sequence": sequence,
            "n_passages": count,
            "passage_fraction": count / total,
        }
        for rank, (sequence, count) in enumerate(counter.most_common(), start=1)
    ]


def pair_lookup(path: Path) -> dict[tuple[str, int], dict]:
    rows = read_csv(path)
    return {
        (row["passage_id"], int(row["capture_index"])): row
        for row in rows
    }


def background_window_rows(
    indexes: dict[str, list[dict]], pairs: dict[tuple[str, int], dict]
) -> list[dict]:
    output = []
    for passage_id, rows in indexes.items():
        first_partial = landmark_index(rows, "parcial", True)
        first_suited = landmark_index(rows, "suited", True)
        last_suited = landmark_index(rows, "suited", False)
        last_partial = landmark_index(rows, "parcial", False)
        anchors = {
            "first_partial": first_partial,
            "first_suited": first_suited,
            "last_suited": last_suited,
            "last_partial": last_partial,
        }
        for index, row in enumerate(rows):
            if row["label"] != "background":
                continue
            pair = pairs.get((passage_id, index + 1))
            for anchor_name, anchor_index in anchors.items():
                if anchor_index is None:
                    continue
                frame_offset = index - anchor_index
                time_offset = float(row["relative_time_ms"]) - float(
                    rows[anchor_index]["relative_time_ms"]
                )
                relevant_side = (
                    anchor_name.startswith("first") and frame_offset < 0
                ) or (anchor_name.startswith("last") and frame_offset > 0)
                if not relevant_side:
                    continue
                if abs(frame_offset) > 10 and abs(time_offset) > 1000:
                    continue
                previous_label = rows[index - 1]["label"] if index > 0 else None
                pdi_valid = previous_label is not None and previous_label != "ruido"
                output.append(
                    {
                        "passage_id": passage_id,
                        "capture_index": index + 1,
                        "anchor": anchor_name,
                        "anchor_capture_index": anchor_index + 1,
                        "frame_offset": frame_offset,
                        "time_offset_ms": time_offset,
                        "relative_time_ms": float(row["relative_time_ms"]),
                        "depth_filename": row["depth_filename"],
                        "rgb_filename": row.get("rgb_filename"),
                        "previous_label": previous_label,
                        "pdi_temporal_valid_after_quality_reset": pdi_valid,
                        "pdi_score": (
                            None
                            if pair is None or not pdi_valid
                            else float(pair[BEST_PDI_FEATURE])
                        ),
                        "pdi_active_at_exploratory_threshold": (
                            None
                            if pair is None or not pdi_valid
                            else float(pair[BEST_PDI_FEATURE])
                            >= BEST_PDI_EXPLORATORY_THRESHOLD
                        ),
                    }
                )
    return output


def quantile_or_none(values: list[float], q: float) -> float | None:
    return None if not values else float(np.quantile(np.asarray(values), q))


def window_summary(rows: list[dict]) -> list[dict]:
    output = []
    for anchor in ("first_partial", "first_suited", "last_suited", "last_partial"):
        selected = [row for row in rows if row["anchor"] == anchor]
        offsets = range(-10, 0) if anchor.startswith("first") else range(1, 11)
        for offset in offsets:
            group = [row for row in selected if row["frame_offset"] == offset]
            scores = [float(row["pdi_score"]) for row in group if row["pdi_score"] is not None]
            output.append(
                {
                    "anchor": anchor,
                    "window_type": "exact_frame_offset",
                    "window": offset,
                    "n_background_frames": len(group),
                    "n_passages": len({row["passage_id"] for row in group}),
                    "pdi_valid_frames": len(scores),
                    "pdi_score_median": quantile_or_none(scores, 0.5),
                    "pdi_score_p75": quantile_or_none(scores, 0.75),
                    "pdi_active_fraction": (
                        None
                        if not scores
                        else sum(score >= BEST_PDI_EXPLORATORY_THRESHOLD for score in scores)
                        / len(scores)
                    ),
                }
            )

        bins = (
            [(-1000, -750), (-750, -500), (-500, -300), (-300, -200), (-200, -100), (-100, 0)]
            if anchor.startswith("first")
            else [(0, 100), (100, 200), (200, 300), (300, 500), (500, 750), (750, 1000)]
        )
        for low, high in bins:
            group = [
                row
                for row in selected
                if low <= float(row["time_offset_ms"]) < high
            ]
            scores = [float(row["pdi_score"]) for row in group if row["pdi_score"] is not None]
            output.append(
                {
                    "anchor": anchor,
                    "window_type": "time_bin_ms",
                    "window": f"[{low},{high})",
                    "n_background_frames": len(group),
                    "n_passages": len({row["passage_id"] for row in group}),
                    "pdi_valid_frames": len(scores),
                    "pdi_score_median": quantile_or_none(scores, 0.5),
                    "pdi_score_p75": quantile_or_none(scores, 0.75),
                    "pdi_active_fraction": (
                        None
                        if not scores
                        else sum(score >= BEST_PDI_EXPLORATORY_THRESHOLD for score in scores)
                        / len(scores)
                    ),
                }
            )
    return output


def relevant_distance(rows: list[dict], index: int) -> tuple[int, float]:
    relevant = [i for i, row in enumerate(rows) if row["label"] in RELEVANT_LABELS]
    frame_distance = min(abs(index - i) for i in relevant)
    timestamp = float(rows[index]["relative_time_ms"])
    time_distance = min(
        abs(timestamp - float(rows[i]["relative_time_ms"])) for i in relevant
    )
    return frame_distance, time_distance


def relabeling_scope(indexes: dict[str, list[dict]]) -> list[dict]:
    output = []
    background_total = sum(
        row["label"] == "background" for rows in indexes.values() for row in rows
    )
    for limit in (1, 2, 3, 5, 10):
        selected = []
        for passage_id, rows in indexes.items():
            for index, row in enumerate(rows):
                if row["label"] != "background":
                    continue
                distance, _ = relevant_distance(rows, index)
                if distance <= limit:
                    selected.append((passage_id, index))
        output.append(
            {
                "scope": "around_any_partial_or_suited",
                "window_type": "frame_distance",
                "window_limit": limit,
                "n_background_frames_to_review": len(selected),
                "fraction_of_all_background": len(selected) / background_total,
                "n_passages": len({passage for passage, _ in selected}),
            }
        )
    for limit in (100, 200, 300, 500, 1000):
        selected = []
        for passage_id, rows in indexes.items():
            for index, row in enumerate(rows):
                if row["label"] != "background":
                    continue
                _, distance = relevant_distance(rows, index)
                if distance <= limit:
                    selected.append((passage_id, index))
        output.append(
            {
                "scope": "around_any_partial_or_suited",
                "window_type": "time_distance_ms",
                "window_limit": limit,
                "n_background_frames_to_review": len(selected),
                "fraction_of_all_background": len(selected) / background_total,
                "n_passages": len({passage for passage, _ in selected}),
            }
        )
    return output


def directed_relabeling_scope(
    indexes: dict[str, list[dict]], pairs: dict[tuple[str, int], dict]
) -> list[dict]:
    within_three: set[tuple[str, int]] = set()
    high_pdi_near: set[tuple[str, int]] = set()
    high_pdi_anywhere: set[tuple[str, int]] = set()
    for passage_id, rows in indexes.items():
        relevant = [i for i, row in enumerate(rows) if row["label"] in RELEVANT_LABELS]
        for index, row in enumerate(rows):
            if row["label"] != "background":
                continue
            key = (passage_id, index + 1)
            frame_distance = min(abs(index - other) for other in relevant)
            time_distance = min(
                abs(float(row["relative_time_ms"]) - float(rows[other]["relative_time_ms"]))
                for other in relevant
            )
            pair = pairs.get(key)
            temporal_valid = index > 0 and rows[index - 1]["label"] != "ruido"
            high_pdi = (
                temporal_valid
                and pair is not None
                and float(pair[BEST_PDI_FEATURE]) >= BEST_PDI_EXPLORATORY_THRESHOLD
            )
            if frame_distance <= 3:
                within_three.add(key)
            if high_pdi and time_distance <= 1000:
                high_pdi_near.add(key)
            if high_pdi:
                high_pdi_anywhere.add(key)
    definitions = {
        "background_within_3_frames_of_relevant": within_three,
        "high_pdi_background_within_1000ms_of_relevant": high_pdi_near,
        "directed_union_3frames_plus_high_pdi_1000ms": within_three | high_pdi_near,
        "directed_union_3frames_plus_all_high_pdi": within_three | high_pdi_anywhere,
    }
    return [
        {
            "scope": name,
            "n_background_frames_to_review": len(keys),
            "fraction_of_all_background": len(keys) / 8859,
            "n_passages": len({passage_id for passage_id, _ in keys}),
        }
        for name, keys in definitions.items()
    ]


def target_distributions(indexes: dict[str, list[dict]]) -> list[dict]:
    output = []
    definitions: list[tuple[str, str, float | None]] = [("target_a_literal", "literal", None)]
    definitions += [(f"expanded_{n}_frames", "frames", float(n)) for n in (1, 2, 3, 5, 10)]
    definitions += [(f"expanded_{n}ms", "time", float(n)) for n in (100, 200, 300, 500, 1000)]
    for name, kind, limit in definitions:
        positive = negative = invalid = 0
        positive_ms = negative_ms = invalid_ms = 0.0
        positive_passages = set()
        for passage_id, rows in indexes.items():
            for index, row in enumerate(rows):
                timestamp = float(row["relative_time_ms"])
                next_timestamp = (
                    float(rows[index + 1]["relative_time_ms"])
                    if index + 1 < len(rows)
                    else timestamp
                )
                duration = max(0.0, next_timestamp - timestamp)
                if row["label"] == "ruido":
                    invalid += 1
                    invalid_ms += duration
                    continue
                is_positive = row["label"] in RELEVANT_LABELS
                if row["label"] == "background" and kind != "literal":
                    frame_distance, time_distance = relevant_distance(rows, index)
                    is_positive = (
                        frame_distance <= int(limit)
                        if kind == "frames"
                        else time_distance <= float(limit)
                    )
                if is_positive:
                    positive += 1
                    positive_ms += duration
                    positive_passages.add(passage_id)
                else:
                    negative += 1
                    negative_ms += duration
        output.append(
            {
                "target_candidate": name,
                "positive_frames": positive,
                "negative_frames": negative,
                "invalid_frames": invalid,
                "positive_to_negative_ratio": positive / negative,
                "positive_fraction_of_valid_frames": positive / (positive + negative),
                "positive_passages": len(positive_passages),
                "positive_duration_ms": positive_ms,
                "negative_duration_ms": negative_ms,
                "invalid_duration_ms": invalid_ms,
                "positive_time_fraction_of_valid": positive_ms / (positive_ms + negative_ms),
            }
        )
    return output


def roi_b_temporal_signal_audit(pair_rows: list[dict]) -> list[dict]:
    roi_area = (0.70 - 0.30) * (0.80 - 0.20)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pair_rows:
        if row["label"] not in RELEVANT_LABELS:
            continue
        if row["previous_label"] == "ruido" or row["label"] == "ruido":
            continue
        global_ratio = float(row["global__changed_ratio_200mm"])
        roi_ratio = float(row["roi_b_y30_70_x20_80__changed_ratio_200mm"])
        global_changed = global_ratio
        roi_changed_global_fraction = roi_ratio * roi_area
        fraction_of_changed_inside = (
            roi_changed_global_fraction / global_changed if global_changed > 0 else 0.0
        )
        enriched = {
            **row,
            "fraction_of_global_changed_pixels_inside_roi_b": fraction_of_changed_inside,
        }
        groups["all_partial_or_suited"].append(enriched)
        groups[f"label_{row['label']}"].append(enriched)
        if row["previous_label"] == "background" and row["label"] == "parcial":
            groups["background_to_partial"].append(enriched)

    output = []
    for group, rows in groups.items():
        values = np.asarray(
            [row["fraction_of_global_changed_pixels_inside_roi_b"] for row in rows],
            dtype=float,
        )
        roi_changed = np.asarray(
            [float(row["roi_b_y30_70_x20_80__changed_ratio_200mm"]) for row in rows],
            dtype=float,
        )
        output.append(
            {
                "group": group,
                "n_pairs": len(rows),
                "roi_share_of_global_change_median": float(np.median(values)),
                "roi_share_of_global_change_p25": float(np.quantile(values, 0.25)),
                "roi_share_of_global_change_p75": float(np.quantile(values, 0.75)),
                "fraction_with_at_least_25pct_change_inside_roi": float(np.mean(values >= 0.25)),
                "fraction_with_at_least_50pct_change_inside_roi": float(np.mean(values >= 0.50)),
                "roi_changed_ratio_median": float(np.median(roi_changed)),
                "fraction_with_roi_changed_ratio_below_1pct": float(np.mean(roi_changed < 0.01)),
            }
        )
    return output


def evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    if len(rows) <= count:
        return rows
    rows = sorted(
        rows,
        key=lambda row: (
            -1.0 if row["pdi_score"] is None else float(row["pdi_score"]),
            row["passage_id"],
            int(row["capture_index"]),
        ),
    )
    indices = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[index] for index in indices]


def manual_sample(window_rows: list[dict]) -> list[dict]:
    definitions = [
        ("pre_partial_minus1", "first_partial", lambda row: row["frame_offset"] == -1, 10),
        ("pre_partial_minus2", "first_partial", lambda row: row["frame_offset"] == -2, 10),
        ("pre_partial_minus3", "first_partial", lambda row: row["frame_offset"] == -3, 10),
        ("pre_partial_approx300ms", "first_partial", lambda row: -350 <= row["time_offset_ms"] < -250, 10),
        ("pre_partial_approx500ms", "first_partial", lambda row: -600 <= row["time_offset_ms"] < -400, 10),
        ("post_partial_plus1", "last_partial", lambda row: row["frame_offset"] == 1, 6),
        ("post_partial_plus2", "last_partial", lambda row: row["frame_offset"] == 2, 6),
        ("post_partial_plus3", "last_partial", lambda row: row["frame_offset"] == 3, 6),
        ("post_partial_approx300ms", "last_partial", lambda row: 250 <= row["time_offset_ms"] < 350, 6),
        ("post_partial_approx500ms", "last_partial", lambda row: 400 <= row["time_offset_ms"] < 600, 6),
    ]
    selected: list[dict] = []
    used: set[tuple[str, int]] = set()
    for stratum, anchor, predicate, count in definitions:
        candidates = [
            row
            for row in window_rows
            if row["anchor"] == anchor
            and predicate(row)
            and (row["passage_id"], int(row["capture_index"])) not in used
        ]
        for row in evenly_spaced(candidates, count):
            key = (row["passage_id"], int(row["capture_index"]))
            used.add(key)
            sample_id = f"S{len(selected) + 1:03d}"
            manual = MANUAL_CLASSIFICATIONS.get(sample_id, (None, None, None))
            selected.append(
                {
                    "sample_id": sample_id,
                    "stratum": stratum,
                    **row,
                    "manual_presence_category": manual[0],
                    "manual_roi_b_coverage": manual[1],
                    "manual_notes": manual[2],
                }
            )
    high_pdi_candidates = sorted(
        [
            row
            for row in window_rows
            if row["pdi_score"] is not None
            and abs(float(row["time_offset_ms"])) <= 1000
            and (row["passage_id"], int(row["capture_index"])) not in used
        ],
        key=lambda row: float(row["pdi_score"]),
        reverse=True,
    )
    high_pdi = []
    for row in high_pdi_candidates:
        key = (row["passage_id"], int(row["capture_index"]))
        if key in used:
            continue
        high_pdi.append(row)
        used.add(key)
        if len(high_pdi) == 20:
            break
    for row in high_pdi:
        key = (row["passage_id"], int(row["capture_index"]))
        sample_id = f"S{len(selected) + 1:03d}"
        manual = MANUAL_CLASSIFICATIONS.get(sample_id, (None, None, None))
        selected.append(
            {
                "sample_id": sample_id,
                "stratum": "high_pdi_background_near_boundary",
                **row,
                "manual_presence_category": manual[0],
                "manual_roi_b_coverage": manual[1],
                "manual_notes": manual[2],
            }
        )
    return selected


def depth_with_roi(depth: np.ndarray) -> Image.Image:
    image = base.depth_preview(depth)
    draw = ImageDraw.Draw(image)
    y_slice, x_slice = base.roi_slices(depth.shape, ROI_B)
    draw.rectangle(
        (x_slice.start, y_slice.start, x_slice.stop - 1, y_slice.stop - 1),
        outline=(255, 0, 0),
        width=2,
    )
    return image


def rgb_with_roi(path: Path) -> Image.Image:
    if path.is_file():
        with Image.open(path) as source:
            image = source.convert("RGB").resize((320, 240))
    else:
        image = Image.new("RGB", (320, 240), "white")
        draw = ImageDraw.Draw(image)
        draw.text((80, 110), "RGB indisponivel", fill="black")
    draw = ImageDraw.Draw(image)
    height, width = 240, 320
    y0, y1, x0, x1 = ROI_B
    draw.rectangle(
        (round(x0 * width), round(y0 * height), round(x1 * width) - 1, round(y1 * height) - 1),
        outline=(255, 0, 0),
        width=2,
    )
    return image


def save_sample_panels(
    data_root: Path,
    indexes: dict[str, list[dict]],
    sample: list[dict],
    output_dir: Path,
) -> list[dict]:
    panel_dir = output_dir / "manual_review_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height = 640, 286
    per_sheet = 8
    output = []
    for sheet_index in range(math.ceil(len(sample) / per_sheet)):
        subset = sample[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        panel = Image.new("RGB", (tile_width * 2, tile_height * 4), "white")
        draw = ImageDraw.Draw(panel)
        for position, row in enumerate(subset):
            x = (position % 2) * tile_width
            y = (position // 2) * tile_height
            passage = row["passage_id"]
            depth = base.read_depth(
                data_root / "DEPTH" / passage / row["depth_filename"]
            )
            anchor_row = indexes[passage][int(row["anchor_capture_index"]) - 1]
            anchor_depth = base.read_depth(
                data_root / "DEPTH" / passage / anchor_row["depth_filename"]
            )
            dep = depth_with_roi(depth)
            anchor_image = depth_with_roi(anchor_depth)
            panel.paste(dep, (x, y + 46))
            panel.paste(anchor_image, (x + 320, y + 46))
            score = "NA" if row["pdi_score"] is None else f"{float(row['pdi_score']):.3f}"
            line1 = (
                f"{row['sample_id']} {row['stratum']} | {passage} "
                f"#{row['capture_index']} | dt={float(row['time_offset_ms']):+.0f}ms "
                f"df={int(row['frame_offset']):+d}"
            )
            line2 = f"background atual | anchor {row['anchor']} | ROI B vermelha | PDI={score}"
            draw.text((x + 4, y + 4), line1, fill="black")
            draw.text((x + 4, y + 23), line2, fill="black")
        path = panel_dir / f"manual_review_{sheet_index + 1:02d}.png"
        panel.save(path)
        for row in subset:
            output.append(
                {
                    "sample_id": row["sample_id"],
                    "panel_path": str(path.relative_to(REPO_ROOT)),
                }
            )
    return output


def manual_summary(sample: list[dict]) -> list[dict]:
    classified = [row for row in sample if row["manual_presence_category"]]
    if not classified:
        return []
    output = []
    groups = {"all": classified}
    groups.update(
        {
            stratum: [row for row in classified if row["stratum"] == stratum]
            for stratum in sorted({row["stratum"] for row in classified})
        }
    )
    groups["pre_entry_all"] = [row for row in classified if row["stratum"].startswith("pre_")]
    groups["post_exit_all"] = [row for row in classified if row["stratum"].startswith("post_")]
    for group, rows in groups.items():
        counts = Counter(row["manual_presence_category"] for row in rows)
        output.append(
            {
                "group": group,
                "n_reviewed": len(rows),
                "clearly_empty": counts["clearly_empty"],
                "ambiguous_possible_animal": counts["ambiguous_possible_animal"],
                "clearly_animal_visible": counts["clearly_animal_visible"],
                "clearly_empty_fraction": counts["clearly_empty"] / len(rows),
                "ambiguous_fraction": counts["ambiguous_possible_animal"] / len(rows),
                "animal_visible_fraction": counts["clearly_animal_visible"] / len(rows),
                "ambiguous_or_visible_fraction": (
                    counts["ambiguous_possible_animal"] + counts["clearly_animal_visible"]
                )
                / len(rows),
            }
        )
    return output


def cohort_summary(
    indexes: dict[str, list[dict]], landmarks: list[dict], frames: list[dict]
) -> dict:
    labels = Counter(row["label"] for row in frames)
    observed = (len(indexes), len(frames), labels["suited"])
    if observed != EXPECTED_COHORT:
        raise ValueError(f"cohort mismatch: {observed} != {EXPECTED_COHORT}")
    return {
        "n_passages": len(indexes),
        "n_frames": len(frames),
        "label_counts": dict(labels),
        "recording_date_counts": dict(Counter(row["recording_date"] for row in landmarks)),
        "quality_gate_temporal_semantics": (
            "ruido clears previous_raw; first subsequent valid frame is a baseline; "
            "only the following valid frame produces a temporal comparison"
        ),
        "roi_b": {"y": [0.30, 0.70], "x": [0.20, 0.80]},
        "best_pdi_feature_from_previous_audit": BEST_PDI_FEATURE,
        "best_pdi_exploratory_threshold": BEST_PDI_EXPLORATORY_THRESHOLD,
        "manual_review_classified": len(MANUAL_CLASSIFICATIONS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--cohort-metrics", type=Path, default=base.DEFAULT_COHORT_METRICS)
    parser.add_argument("--pair-features", type=Path, default=DEFAULT_PAIR_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort = base.load_cohort(args.cohort_metrics)
    indexes = base.load_indexes(args.data_root, cohort)
    pairs = pair_lookup(args.pair_features)
    pair_rows = list(pairs.values())
    frames = build_frame_rows(indexes)
    landmarks = passage_landmarks(indexes)
    patterns = pattern_summary(landmarks, "sequence_with_noise")
    patterns += pattern_summary(landmarks, "sequence_valid_only")
    windows = background_window_rows(indexes, pairs)
    sample = manual_sample(windows)
    panel_rows = save_sample_panels(args.data_root, indexes, sample, args.output_dir)
    panel_by_sample = {row["sample_id"]: row["panel_path"] for row in panel_rows}
    for row in sample:
        row["panel_path"] = panel_by_sample[row["sample_id"]]

    write_csv(args.output_dir / "passage_landmarks.csv", landmarks)
    write_csv(args.output_dir / "label_sequence_patterns.csv", patterns)
    write_csv(args.output_dir / "background_relative_windows.csv", windows)
    write_csv(args.output_dir / "background_window_summary.csv", window_summary(windows))
    write_csv(
        args.output_dir / "relabeling_scope.csv",
        relabeling_scope(indexes) + directed_relabeling_scope(indexes, pairs),
    )
    write_csv(args.output_dir / "target_candidate_distribution.csv", target_distributions(indexes))
    write_csv(args.output_dir / "roi_b_temporal_signal_audit.csv", roi_b_temporal_signal_audit(pair_rows))
    write_csv(args.output_dir / "manual_review_sample.csv", sample)
    write_csv(args.output_dir / "manual_review_summary.csv", manual_summary(sample))
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(cohort_summary(indexes, landmarks, frames), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
