#!/usr/bin/env python3
"""Calibra offline MAD/histerese contra o ground truth humano ``suited``.

O cohort operacional padrao e definido pelas 184 chaves ``animals`` da run
fixa publicada. O script percorre todos os frames originais dessas passagens;
nao executa PADE, selector ou qualquer modelo.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import skimage.io as ski


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domain.visual_activity import VisualActivityDetector, VisualState  # noqa: E402
from domain.visual_activity import mean_absolute_depth_difference  # noqa: E402


DEFAULT_COHORT_METRICS = (
    "power_runs/battery_mas-single_20260708_104924/"
    "mas-single_1fps_r1/**/metrics.json"
)
EXPECTED_COHORT = (184, 13_741, 1_655)
ALLOWED_LABELS = {"suited", "background", "parcial", "ruido"}
QUANTILES = (0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
QUANTILE_NAMES = ("min", "median", "p75", "p90", "p95", "p99", "max")
DEFAULT_PATIENCE = (1, 2, 3, 5)


def _resolve_one(pattern: str) -> Path:
    candidate = Path(pattern)
    absolute = str(candidate if candidate.is_absolute() else REPO_ROOT / candidate)
    files = [Path(path) for path in glob.glob(absolute, recursive=True) if Path(path).is_file()]
    if len(files) != 1:
        raise ValueError(
            f"expected one cohort metrics file for {pattern!r}; found {len(files)}"
        )
    return files[0]


def load_operational_cohort(pattern: str = DEFAULT_COHORT_METRICS) -> list[str]:
    path = _resolve_one(pattern)
    with path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    animals = metrics.get("animals")
    if not isinstance(animals, dict):
        raise ValueError(f"invalid cohort metrics: {path}")
    return sorted(animals)


def load_indexes(data_root: Path, passage_ids: Iterable[str]) -> dict[str, list[dict]]:
    indexes: dict[str, list[dict]] = {}
    for passage_id in sorted(passage_ids):
        path = data_root / "animal-tags" / passage_id / "simulation_index.json"
        if not path.is_file():
            raise ValueError(f"missing passage index: {path}")
        with path.open(encoding="utf-8") as file:
            frames = json.load(file)
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"empty or invalid passage index: {path}")
        frames.sort(key=lambda frame: float(frame["relative_time_ms"]))
        for frame in frames:
            label = frame.get("label")
            if label not in ALLOWED_LABELS:
                raise ValueError(
                    f"unknown label in {passage_id}/{frame.get('depth_filename')}: {label!r}"
                )
            depth = data_root / "DEPTH" / passage_id / frame["depth_filename"]
            if not depth.is_file():
                raise ValueError(f"missing depth frame: {depth}")
        indexes[passage_id] = frames
    return indexes


def collect_mad_records(
    data_root: Path,
    indexes: dict[str, list[dict]],
) -> list[dict]:
    records: list[dict] = []
    for passage_id in sorted(indexes):
        previous = None
        previous_timestamp = None
        for capture_index, frame in enumerate(indexes[passage_id], start=1):
            raw = ski.imread(
                data_root / "DEPTH" / passage_id / frame["depth_filename"]
            )
            timestamp = float(frame["relative_time_ms"])
            mad = (
                None
                if previous is None
                else mean_absolute_depth_difference(previous, raw)
            )
            records.append(
                {
                    "passage_id": passage_id,
                    "capture_index": capture_index,
                    "timestamp_ms": timestamp,
                    "delta_t_ms": (
                        None
                        if previous_timestamp is None
                        else timestamp - previous_timestamp
                    ),
                    "depth_filename": frame["depth_filename"],
                    "label": frame["label"],
                    "mad": mad,
                }
            )
            previous = raw
            previous_timestamp = timestamp
    return records


def audit_cohort(records: list[dict], expected: tuple[int, int, int] | None = EXPECTED_COHORT) -> dict:
    audit = {
        "n_passages": len({record["passage_id"] for record in records}),
        "n_frames": len(records),
        "n_suited_frames": sum(record["label"] == "suited" for record in records),
    }
    if expected is not None:
        observed = tuple(audit.values())
        if observed != expected:
            raise ValueError(f"operational cohort mismatch: {observed} != {expected}")
    return audit


def mad_distribution(records: list[dict]) -> list[dict]:
    groups = {
        "global": lambda _: True,
        "suited": lambda row: row["label"] == "suited",
        "non-suited": lambda row: row["label"] != "suited",
        "background": lambda row: row["label"] == "background",
        "parcial": lambda row: row["label"] == "parcial",
        "ruido": lambda row: row["label"] == "ruido",
    }
    output = []
    for name, predicate in groups.items():
        values = np.asarray(
            [row["mad"] for row in records if row["mad"] is not None and predicate(row)],
            dtype=float,
        )
        quantiles = (
            np.quantile(values, QUANTILES)
            if values.size
            else [None] * len(QUANTILES)
        )
        output.append(
            {
                "group": name,
                "n_pairs": int(values.size),
                **dict(zip(QUANTILE_NAMES, quantiles)),
            }
        )
    return output


def threshold_candidates(records: list[dict]) -> list[float]:
    values = np.asarray([row["mad"] for row in records if row["mad"] is not None])
    # Pequena grade informada pela distribuicao, nao um threshold final.
    return sorted({float(value) for value in np.quantile(values, (0.1, 0.25, 0.5, 0.75, 0.9, 0.95))})


def evaluate_configuration(
    records: list[dict],
    threshold: float,
    patience: int,
) -> tuple[dict, list[dict]]:
    passage_rows: list[dict] = []
    by_passage: dict[str, list[dict]] = {}
    for row in records:
        by_passage.setdefault(row["passage_id"], []).append(row)

    for passage_id, frames in by_passage.items():
        detector = VisualActivityDetector(threshold, patience)
        evaluated = []
        for frame in frames:
            if frame["mad"] is None:
                state = VisualState.IDLE
            else:
                state = detector.observe_mad(frame["mad"]).visual_state
            evaluated.append((frame, state))

        suited = [frame for frame, _ in evaluated if frame["label"] == "suited"]
        active = [frame for frame, state in evaluated if state is VisualState.ACTIVE]
        suited_active = [
            frame
            for frame, state in evaluated
            if frame["label"] == "suited" and state is VisualState.ACTIVE
        ]
        first_suited = suited[0]["timestamp_ms"] if suited else None
        first_active = active[0]["timestamp_ms"] if active else None
        delay = (
            None
            if first_suited is None or first_active is None
            else first_active - first_suited
        )
        passage_rows.append(
            {
                "threshold": threshold,
                "idle_patience_frames": patience,
                "passage_id": passage_id,
                "total_frames": len(frames),
                "suited_frames": len(suited),
                "suited_frames_active": len(suited_active),
                "active_frames": len(active),
                "first_suited_timestamp_ms": first_suited,
                "first_active_timestamp_ms": first_active,
                "activation_delay_ms": delay,
                "missed_suited_frames": len(suited) - len(suited_active),
                "suited_passage_covered": bool(suited_active) if suited else None,
            }
        )

    suited_passages = [row for row in passage_rows if row["suited_frames"] > 0]
    delays = np.asarray(
        [row["activation_delay_ms"] for row in suited_passages if row["activation_delay_ms"] is not None],
        dtype=float,
    )
    missed = [row["passage_id"] for row in suited_passages if not row["suited_passage_covered"]]
    total_suited = sum(row["suited_frames"] for row in passage_rows)
    total_suited_active = sum(row["suited_frames_active"] for row in passage_rows)
    total_frames = sum(row["total_frames"] for row in passage_rows)
    total_active = sum(row["active_frames"] for row in passage_rows)
    summary = {
        "configuration": "mad",
        "threshold": threshold,
        "idle_patience_frames": patience,
        "n_passages": len(passage_rows),
        "n_suited_passages": len(suited_passages),
        "n_suited_passages_covered": len(suited_passages) - len(missed),
        "suited_passage_coverage": (
            (len(suited_passages) - len(missed)) / len(suited_passages)
            if suited_passages else 0.0
        ),
        "suited_frame_retention": total_suited_active / total_suited if total_suited else 0.0,
        "active_ratio": total_active / total_frames if total_frames else 0.0,
        "activation_delay_median_ms": float(np.median(delays)) if delays.size else None,
        "activation_delay_p95_ms": float(np.quantile(delays, 0.95)) if delays.size else None,
        "n_missed_passages": len(missed),
        "missed_passage_ids": ";".join(missed),
    }
    return summary, passage_rows


def always_active_baseline(records: list[dict]) -> dict:
    passage_ids = {row["passage_id"] for row in records}
    suited_passages = {row["passage_id"] for row in records if row["label"] == "suited"}
    return {
        "configuration": "always_active",
        "threshold": None,
        "idle_patience_frames": None,
        "n_passages": len(passage_ids),
        "n_suited_passages": len(suited_passages),
        "n_suited_passages_covered": len(suited_passages),
        "suited_passage_coverage": 1.0,
        "suited_frame_retention": 1.0,
        "active_ratio": 1.0,
        "activation_delay_median_ms": None,
        "activation_delay_p95_ms": None,
        "n_missed_passages": 0,
        "missed_passage_ids": "",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/exp1")
    parser.add_argument("--cohort-metrics", default=DEFAULT_COHORT_METRICS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data-analysis/visual_activity_output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = load_operational_cohort(args.cohort_metrics)
    indexes = load_indexes(args.data_root, cohort)
    records = collect_mad_records(args.data_root, indexes)
    audit = audit_cohort(records)
    distributions = mad_distribution(records)

    summaries = [always_active_baseline(records)]
    passage_rows: list[dict] = []
    for threshold in threshold_candidates(records):
        for patience in DEFAULT_PATIENCE:
            summary, rows = evaluate_configuration(records, threshold, patience)
            summaries.append(summary)
            passage_rows.extend(rows)

    summaries[1:] = sorted(
        summaries[1:],
        key=lambda row: (
            -row["suited_passage_coverage"],
            -row["suited_frame_retention"],
            row["active_ratio"],
        ),
    )
    write_csv(args.output_dir / "mad_distribution.csv", distributions)
    write_csv(args.output_dir / "visual_activity_calibration_summary.csv", summaries)
    write_csv(args.output_dir / "visual_activity_calibration_by_passage.csv", passage_rows)
    write_csv(args.output_dir / "mad_by_frame.csv", records)
    print(json.dumps({"audit": audit, "thresholds": threshold_candidates(records)}, indent=2))


if __name__ == "__main__":
    main()
