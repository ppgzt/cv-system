#!/usr/bin/env python3
"""Auditoria offline do quality gate de ruido do Visual Event.

Le apenas os features ja extraidos do cohort operacional. Nao altera o
detector de producao, o dataset ou qualquer agente PADE.

As regras candidatas usam somente cortes que ja foram publicados na auditoria
anterior de ruido. O objetivo e tornar explicito o custo do corte atual e de
alternativas simples; nao e procurar novos hiperparametros.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = (
    REPO_ROOT / "data-analysis/visual_event_noise_pdi_output/frame_features.csv"
)
OUTPUT_DIR = REPO_ROOT / "data-analysis/visual_event_quality_gate_audit/output"

# Thresholds preexistentes em noise_single_frame_performance.csv.
P99_YOUDEN = 2230.0
P99_RECALL95 = 2268.01
FRACTION_GE_2500_YOUDEN = 0.0027473958333333335
FRACTION_GE_2500_RECALL95 = 0.0034505208333333332


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in (
            "capture_index",
            "relative_time_ms",
            "elapsed_from_passage_start_ms",
            "depth_p95_mm",
            "depth_p99_mm",
            "fraction_ge_2500mm",
            "fraction_equal_2600mm",
            "histogram_dominant_bin_fraction",
        ):
            row[field] = float(row[field])
        row["capture_index"] = int(row["capture_index"])
    return rows


def candidate_rules(row: dict) -> dict[str, bool]:
    p99 = row["depth_p99_mm"]
    high_fraction = row["fraction_ge_2500mm"]
    return {
        "current_p99_youden": p99 >= P99_YOUDEN,
        "p99_recall95": p99 >= P99_RECALL95,
        "fraction_ge_2500_youden": high_fraction >= FRACTION_GE_2500_YOUDEN,
        "fraction_ge_2500_recall95": high_fraction >= FRACTION_GE_2500_RECALL95,
        "p99_and_fraction_youden": (
            p99 >= P99_YOUDEN and high_fraction >= FRACTION_GE_2500_YOUDEN
        ),
    }


def metrics(rows: list[dict], rule_name: str) -> dict:
    tp = fp = tn = fn = 0
    by_label = Counter()
    for row in rows:
        predicted_invalid = candidate_rules(row)[rule_name]
        actual_noise = row["label"] == "ruido"
        if predicted_invalid:
            by_label[row["label"]] += 1
        if predicted_invalid and actual_noise:
            tp += 1
        elif predicted_invalid:
            fp += 1
        elif actual_noise:
            fn += 1
        else:
            tn += 1
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    precision = tp / (tp + fp) if tp + fp else None
    return {
        "rule": rule_name,
        "invalid_count": tp + fp,
        "true_noise_invalid": tp,
        "non_noise_invalid": fp,
        "noise_not_invalid": fn,
        "true_valid": tn,
        "noise_recall": recall,
        "non_noise_false_positive_rate": fpr,
        "invalid_precision_noise": precision,
        "invalid_background": by_label["background"],
        "invalid_parcial": by_label["parcial"],
        "invalid_suited": by_label["suited"],
        "invalid_ruido": by_label["ruido"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = read_rows(FEATURES_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rules = list(candidate_rules(rows[0]))
    summary = [metrics(rows, rule) for rule in rules]
    write_csv(OUTPUT_DIR / "candidate_rule_summary.csv", summary)

    current = []
    for row in rows:
        flags = candidate_rules(row)
        if flags["current_p99_youden"]:
            current.append(
                {
                    "passage_id": row["passage_id"],
                    "capture_index": row["capture_index"],
                    "relative_time_ms": row["relative_time_ms"],
                    "elapsed_from_passage_start_ms": row["elapsed_from_passage_start_ms"],
                    "label": row["label"],
                    "depth_filename": row["depth_filename"],
                    "depth_p95_mm": row["depth_p95_mm"],
                    "depth_p99_mm": row["depth_p99_mm"],
                    "fraction_ge_2500mm": row["fraction_ge_2500mm"],
                    "fraction_equal_2600mm": row["fraction_equal_2600mm"],
                    "histogram_dominant_bin_fraction": row[
                        "histogram_dominant_bin_fraction"
                    ],
                    "current_gate_result": "INVALID",
                    "human_noise": row["label"] == "ruido",
                    "current_outcome": (
                        "true_noise" if row["label"] == "ruido" else "false_positive"
                    ),
                    "fraction_youden_result": (
                        "INVALID"
                        if flags["fraction_ge_2500_youden"]
                        else "VALID"
                    ),
                    "conjunctive_result": (
                        "INVALID"
                        if flags["p99_and_fraction_youden"]
                        else "VALID"
                    ),
                }
            )
    current.sort(key=lambda r: (r["current_outcome"], r["label"], r["passage_id"], r["capture_index"]))
    write_csv(OUTPUT_DIR / "current_gate_all_invalid_frames.csv", current)

    # Passage-level concentration makes manual inspection of false positives tractable.
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in current:
        grouped[(row["current_outcome"], row["label"])].append(row)
    passage_summary = []
    for (outcome, label), selected in sorted(grouped.items()):
        by_passage: dict[str, list[dict]] = defaultdict(list)
        for row in selected:
            by_passage[row["passage_id"]].append(row)
        for passage_id, passage_rows in sorted(by_passage.items()):
            passage_summary.append(
                {
                    "current_outcome": outcome,
                    "label": label,
                    "passage_id": passage_id,
                    "count": len(passage_rows),
                    "first_capture_index": min(r["capture_index"] for r in passage_rows),
                    "last_capture_index": max(r["capture_index"] for r in passage_rows),
                    "min_p99_mm": min(r["depth_p99_mm"] for r in passage_rows),
                    "max_p99_mm": max(r["depth_p99_mm"] for r in passage_rows),
                    "fraction_youden_keeps_invalid": sum(
                        r["fraction_youden_result"] == "INVALID" for r in passage_rows
                    ),
                    "conjunctive_keeps_invalid": sum(
                        r["conjunctive_result"] == "INVALID" for r in passage_rows
                    ),
                }
            )
    write_csv(OUTPUT_DIR / "current_gate_invalids_by_passage.csv", passage_summary)

    # Explicitly expose all human ruido missed by the current rule.
    missed = [
        {
            "passage_id": row["passage_id"],
            "capture_index": row["capture_index"],
            "relative_time_ms": row["relative_time_ms"],
            "label": row["label"],
            "depth_filename": row["depth_filename"],
            "depth_p99_mm": row["depth_p99_mm"],
            "fraction_ge_2500mm": row["fraction_ge_2500mm"],
        }
        for row in rows
        if row["label"] == "ruido" and not candidate_rules(row)["current_p99_youden"]
    ]
    write_csv(OUTPUT_DIR / "human_noise_missed_by_current_gate.csv", missed)

    payload = {
        "scope": {
            "total_frames": len(rows),
            "human_noise_frames": sum(row["label"] == "ruido" for row in rows),
            "cohort": "operational 184 passages",
        },
        "rules": {
            "current_p99_youden": f"depth_p99_mm >= {P99_YOUDEN}",
            "p99_recall95": f"depth_p99_mm >= {P99_RECALL95}",
            "fraction_ge_2500_youden": (
                f"fraction_ge_2500mm >= {FRACTION_GE_2500_YOUDEN}"
            ),
            "fraction_ge_2500_recall95": (
                f"fraction_ge_2500mm >= {FRACTION_GE_2500_RECALL95}"
            ),
            "p99_and_fraction_youden": (
                f"depth_p99_mm >= {P99_YOUDEN} AND "
                f"fraction_ge_2500mm >= {FRACTION_GE_2500_YOUDEN}"
            ),
        },
        "notes": [
            "Candidate thresholds were copied from the earlier single-frame noise audit.",
            "This report is diagnostic only; it does not modify the production quality gate.",
            "The human label ruido is the positive class for this audit.",
        ],
    }
    (OUTPUT_DIR / "audit_configuration.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
