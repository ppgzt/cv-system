#!/usr/bin/env python3
"""Review depth evidence and rerun only the baseline PDI with a conjunctive gate.

This is an offline diagnostic. It does not change VisualEventAgent or the
runtime rule. The PDI score, ROI, 200 mm pixel threshold, threshold-selection
protocol and hysteresis are imported unchanged from the prior ablation.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "data-analysis", REPO_ROOT / "data-analysis/visual_event_preprocessing_ablation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import visual_event_diagnostic as base  # noqa: E402
import run_ablation as ablation  # noqa: E402

from audit_quality_gate import (  # noqa: E402
    FEATURES_CSV,
    FRACTION_GE_2500_YOUDEN,
    OUTPUT_DIR,
    P99_YOUDEN,
    candidate_rules,
    read_rows,
    write_csv,
)


DATA_ROOT = base.DEFAULT_DATA_ROOT
COHORT_METRICS = base.DEFAULT_COHORT_METRICS
REVIEW_DIR = OUTPUT_DIR / "depth_review"
TEMPORAL_FEATURES = (
    REPO_ROOT / "data-analysis/visual_event_classical_pdi_output/temporal_feature_rows.csv"
)


def conjunction_gate(row: dict) -> bool:
    return candidate_rules(row)["p99_and_fraction_youden"]


def existing_score_lookup() -> dict[tuple[str, int], float]:
    """Reuse physical pair scores already materialized by the prior PDI audit."""

    scores: dict[tuple[str, int], float] = {}
    with TEMPORAL_FEATURES.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = row["baseline_component_coherence"]
            if value:
                scores[(row["passage_id"], int(row["capture_index"]))] = float(value)
    return scores


def build_baseline_series(
    indexes: dict[str, list[dict]],
    feature_lookup: dict[tuple[str, int], dict],
    raw_scores: dict[tuple[str, int], float],
    gate_mode: str,
) -> dict[str, list[float]]:
    """Exact reset semantics; only the invalid predicate varies."""

    output: dict[str, list[float]] = {}
    for passage_id, frames in indexes.items():
        previous_valid = False
        values = []
        for capture_index, frame in enumerate(frames, start=1):
            features = feature_lookup[(passage_id, capture_index)]
            if gate_mode == "oracle_label":
                invalid = frame["label"] == "ruido"
            elif gate_mode == "current_p99":
                invalid = features["depth_p99_mm"] >= P99_YOUDEN
            elif gate_mode == "conjunctive":
                invalid = conjunction_gate(features)
            else:
                raise ValueError(gate_mode)
            if invalid:
                previous_valid = False
                values.append(float("nan"))
            elif not previous_valid:
                previous_valid = True
                values.append(float("nan"))
            else:
                key = (passage_id, capture_index)
                score = raw_scores.get(key)
                if score is None:
                    # Only pairs formerly hidden behind an old reset need IO.
                    previous = base.read_depth(
                        DATA_ROOT / "DEPTH" / passage_id / frames[capture_index - 2]["depth_filename"]
                    )
                    current = base.read_depth(
                        DATA_ROOT / "DEPTH" / passage_id / frame["depth_filename"]
                    )
                    score = ablation.pdi_score(
                        previous, current, ablation.PHASE_ONE_VARIANTS["V0_baseline"]
                    )
                    raw_scores[key] = score
                values.append(score)
        output[passage_id] = values
    return output


def operational_details(
    indexes: dict[str, list[dict]], series: dict[str, list[float]], direction: float, threshold: float
) -> tuple[dict, list[dict]]:
    """Same state machine as ablation.operational_metrics plus passage details."""

    details = []
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
                    if no_motion >= ablation.IDLE_PATIENCE:
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
        details.append(
            {
                "passage_id": passage_id,
                "n_frames": len(frames),
                "n_suited": len(suited),
                "n_suited_forward_active": sum(pre_states[index] for index in suited),
                "suited_passage_covered_forward": bool(any(pre_states[index] for index in suited)),
                "first_activation_capture_index": None if first_activation is None else first_activation + 1,
                "first_suited_capture_index": None if first_suited is None else first_suited + 1,
                "first_activation_before_suited": bool(first_activation is not None and first_suited is not None and first_activation < first_suited),
                "first_activation_in_partial": bool(first_activation is not None and labels[first_activation] == "parcial"),
                "activation_delay_to_first_suited_ms": None if first_activation is None or first_suited is None else float(timestamps[first_activation] - timestamps[first_suited]),
                "false_active_background_frames": sum(post_states[index] for index in background),
                "background_frames": len(background),
                "active_time_ms": float(np.sum(intervals[np.asarray(post_states[:-1], dtype=bool)])),
                "total_time_ms": float(np.sum(intervals)),
            }
        )

    suited_rows = [row for row in details if row["n_suited"]]
    total_suited = sum(row["n_suited"] for row in suited_rows)
    total_background = sum(row["background_frames"] for row in details)
    total_time = sum(row["total_time_ms"] for row in details)
    summary = {
        "suited_passage_coverage": sum(row["suited_passage_covered_forward"] for row in suited_rows) / len(suited_rows),
        "suited_retention": sum(row["n_suited_forward_active"] for row in suited_rows) / total_suited,
        "false_active_background": sum(row["false_active_background_frames"] for row in details) / total_background,
        "time_active_ratio": sum(row["active_time_ms"] for row in details) / total_time,
        "activation_before_suited": sum(row["first_activation_before_suited"] for row in suited_rows) / len(suited_rows),
        "activation_in_partial": sum(row["first_activation_in_partial"] for row in suited_rows) / len(suited_rows),
    }
    return summary, details


def depth_thumbnail(frame: np.ndarray, size=(160, 120)) -> Image.Image:
    # Fixed scale makes the comparison within and across panels interpretable.
    pixels = np.clip(frame.astype(np.float32) / 2600.0 * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels).resize(size, Image.Resampling.NEAREST).convert("RGB")


def make_contact_pages(
    selected: list[dict], indexes: dict[str, list[dict]], title: str, prefix: str
) -> list[str]:
    """Create depth-only contextual panels: prev / TARGET / next, no RGB needed."""

    selected_by_passage = defaultdict(dict)
    for row in selected:
        selected_by_passage[row["passage_id"]][int(row["capture_index"])] = row
    pages = []
    per_page, columns = 9, 3
    card_w, card_h = 510, 175
    for page_start in range(0, len(selected), per_page):
        chunk = selected[page_start : page_start + per_page]
        rows_count = math.ceil(len(chunk) / columns)
        image = Image.new("RGB", (columns * card_w, 36 + rows_count * card_h), "white")
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), title, fill="black")
        for pos, entry in enumerate(chunk):
            col, row_index = pos % columns, pos // columns
            x0, y0 = col * card_w, 36 + row_index * card_h
            passage_rows = indexes[entry["passage_id"]]
            target = int(entry["capture_index"]) - 1
            for relative, label in ((-1, "prev"), (0, "TARGET"), (1, "next")):
                at = target + relative
                if not 0 <= at < len(passage_rows):
                    continue
                frame = base.read_depth(
                    DATA_ROOT / "DEPTH" / entry["passage_id"] / passage_rows[at]["depth_filename"]
                )
                tile = depth_thumbnail(frame)
                tx = x0 + (relative + 1) * 165
                image.paste(tile, (tx, y0 + 27))
                color = "red" if relative == 0 else "black"
                draw.text((tx, y0 + 12), f"{label} #{at + 1} {passage_rows[at]['label']}", fill=color)
            draw.rectangle((x0 + 165, y0 + 27, x0 + 325, y0 + 147), outline="red", width=3)
            draw.text(
                (x0, y0 + 150),
                f"{entry['passage_id']} #{entry['capture_index']} | p99={float(entry['depth_p99_mm']):.0f} | frac={float(entry['fraction_ge_2500mm']):.4f}",
                fill="black",
            )
        destination = REVIEW_DIR / f"{prefix}_{page_start // per_page + 1:02d}.png"
        image.save(destination)
        pages.append(str(destination.relative_to(REPO_ROOT)))
    return pages


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    feature_rows = read_rows(FEATURES_CSV)
    features = {(row["passage_id"], row["capture_index"]): row for row in feature_rows}
    all_current_invalid = [row for row in feature_rows if row["depth_p99_mm"] >= P99_YOUDEN]
    false_positives = [
        row for row in all_current_invalid
        if row["label"] != "ruido" and conjunction_gate(row)
    ]
    missed_noise = [
        row for row in feature_rows
        if row["label"] == "ruido" and not conjunction_gate(row)
    ]
    false_positives.sort(key=lambda row: (row["label"], row["passage_id"], row["capture_index"]))
    missed_noise.sort(key=lambda row: (row["passage_id"], row["capture_index"]))

    passage_ids = base.load_cohort(COHORT_METRICS)
    indexes = base.load_indexes(DATA_ROOT, passage_ids)
    fp_pages = make_contact_pages(false_positives, indexes, "Conjunctive-gate false positives (human non-ruido)", "conjunctive_false_positives")
    missed_pages = make_contact_pages(missed_noise, indexes, "Human ruido missed by conjunctive gate", "missed_human_noise")
    write_csv(OUTPUT_DIR / "conjunctive_false_positive_review.csv", false_positives)
    write_csv(OUTPUT_DIR / "conjunctive_missed_human_noise_review.csv", missed_noise)

    # The score itself is exactly V0 from the prior ablation; only reset points change.
    raw_scores = existing_score_lookup()
    oracle = build_baseline_series(indexes, features, raw_scores, "oracle_label")
    current = build_baseline_series(indexes, features, raw_scores, "current_p99")
    conjunctive = build_baseline_series(indexes, features, raw_scores, "conjunctive")
    oracle_metrics = ablation.frame_metrics(indexes, oracle)
    threshold, direction = oracle_metrics["threshold_directed"], oracle_metrics["direction"]

    comparison = []
    for name, series in (("current_p99", current), ("conjunctive", conjunctive)):
        frame = ablation.frame_metrics(indexes, series, direction=direction, threshold_directed=threshold)
        operational, details = operational_details(indexes, series, direction, threshold)
        comparison.append({"quality_gate": name, **frame, **operational})
        write_csv(OUTPUT_DIR / f"baseline_{name}_by_passage.csv", details)
    write_csv(OUTPUT_DIR / "baseline_gate_comparison.csv", comparison)

    # Keep the 0170 result concise and independently inspectable.
    passage_0170 = []
    for name in ("current_p99", "conjunctive"):
        with (OUTPUT_DIR / f"baseline_{name}_by_passage.csv").open(newline="") as handle:
            row = next(item for item in csv.DictReader(handle) if item["passage_id"] == "0170")
            passage_0170.append({"quality_gate": name, **row})
    write_csv(OUTPUT_DIR / "passage_0170_gate_comparison.csv", passage_0170)

    summary = {
        "scope": "offline; baseline PDI only; no production modification",
        "current_gate": f"depth_p99_mm >= {P99_YOUDEN}",
        "conjunctive_gate": (
            f"depth_p99_mm >= {P99_YOUDEN} AND "
            f"fraction_ge_2500mm >= {FRACTION_GE_2500_YOUDEN}"
        ),
        "threshold_selection": "unchanged: oracle-label quality gate then Youden over baseline score",
        "baseline_score": "ROI B, absdiff, 200 mm mask, largest-component changed fraction",
        "false_positive_frames_reviewed": len(false_positives),
        "missed_human_noise_reviewed": len(missed_noise),
        "false_positive_panel_pages": fp_pages,
        "missed_noise_panel_pages": missed_pages,
        "oracle_threshold_directed": threshold,
        "direction": direction,
    }
    (OUTPUT_DIR / "rerun_configuration.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
