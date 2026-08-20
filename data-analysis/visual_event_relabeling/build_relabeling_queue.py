#!/usr/bin/env python3
"""Fila dirigida de relabeling RGB do Visual Event.

Este modulo e exclusivamente offline. Ele usa ``simulation_index.json`` apenas
para ler metadados e nunca abre, procura, baixa ou copia arquivos RGB. A revisao
humana acontece manualmente no Google Drive usando os filenames aqui registrados.

Subcomandos:

* ``build``: cria a fila RGB principal, o piloto e o resumo por passagem;
* ``consolidate``: converte revisoes concluidas em labels separados do dataset;
* ``validate``: valida a fila e os labels sem escrever no dataset original.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_DIR.parents[1]
ANALYSIS_DIR = REPO_ROOT / "data-analysis"
for path in (REPO_ROOT, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import visual_event_diagnostic as base  # noqa: E402
import visual_event_label_audit as label_audit  # noqa: E402


DEFAULT_PAIR_FEATURES = label_audit.DEFAULT_PAIR_FEATURES
PDI_FEATURE = label_audit.BEST_PDI_FEATURE
PDI_THRESHOLD = label_audit.BEST_PDI_EXPLORATORY_THRESHOLD
RELEVANT_LABELS = frozenset({"parcial", "suited"})
EXPECTED_COHORT = (184, 13_741, 1_655)
EXPECTED_SCOPE = {
    "boundary": 1_077,
    "high_pdi": 146,
    "boundary+high_pdi": 78,
}
EXPECTED_CANDIDATES = 1_301
PILOT_PER_REASON = 50
PILOT_SEED = 20260818

FINAL_REVIEW_VALUES = frozenset({"", "CLEAR_EMPTY", "ANIMAL_VISIBLE", "AMBIGUOUS"})
FINAL_TO_TARGET = {
    "CLEAR_EMPTY": "NEGATIVE",
    "ANIMAL_VISIBLE": "POSITIVE",
    "AMBIGUOUS": "EXCLUDE",
}

MANIFEST_FIELDS = [
    "review_order",
    "passage_id",
    "capture_index",
    "relative_time_ms",
    "rgb_filename",
    "rgb_prev_3",
    "rgb_prev_2",
    "rgb_prev_1",
    "rgb_next_1",
    "rgb_next_2",
    "rgb_next_3",
    "original_label",
    "candidate_reason",
    "final_review",
    "notes",
]

PASSAGE_SUMMARY_FIELDS = [
    "passage_id",
    "candidate_count",
    "min_capture_index",
    "max_capture_index",
]

RELABEL_FIELDS = [
    "passage_id",
    "capture_index",
    "relative_time_ms",
    "rgb_filename",
    "original_label",
    "candidate_reason",
    "final_review",
    "visual_event_label",
    "training_eligibility",
    "notes",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_paths(data_root: Path, passage_ids: Iterable[str]) -> dict[str, Path]:
    return {
        passage_id: data_root / "animal-tags" / passage_id / "simulation_index.json"
        for passage_id in passage_ids
    }


def load_operational_indexes(
    data_root: Path, cohort_metrics: Path
) -> tuple[list[str], dict[str, list[dict]], dict[str, str]]:
    passage_ids = base.load_cohort(cohort_metrics)
    paths = index_paths(data_root, passage_ids)
    hashes_before = {passage_id: file_sha256(path) for passage_id, path in paths.items()}
    indexes = base.load_indexes(data_root, passage_ids)
    observed = (
        len(indexes),
        sum(len(rows) for rows in indexes.values()),
        sum(row["label"] == "suited" for rows in indexes.values() for row in rows),
    )
    if observed != EXPECTED_COHORT:
        raise ValueError(f"operational cohort mismatch: {observed} != {EXPECTED_COHORT}")
    missing_rgb_metadata = [
        (passage_id, index + 1)
        for passage_id, rows in indexes.items()
        for index, row in enumerate(rows)
        if not row.get("rgb_filename")
    ]
    if missing_rgb_metadata:
        raise ValueError(f"missing rgb_filename metadata: {missing_rgb_metadata[:10]}")
    return passage_ids, indexes, hashes_before


def relevant_distance(rows: list[dict], index: int) -> tuple[int, float]:
    relevant = [position for position, row in enumerate(rows) if row["label"] in RELEVANT_LABELS]
    if not relevant:
        raise ValueError("passage without parcial/suited cannot enter operational scope")
    frame_distance = min(abs(index - position) for position in relevant)
    timestamp = float(rows[index]["relative_time_ms"])
    time_distance_ms = min(
        abs(timestamp - float(rows[position]["relative_time_ms"]))
        for position in relevant
    )
    return frame_distance, time_distance_ms


def build_candidates(
    indexes: dict[str, list[dict]],
    pairs: dict[tuple[str, int], dict],
) -> list[dict]:
    """Reproduz exatamente o escopo aprovado, sem recalibrar a PDI."""

    output = []
    for passage_id, rows in indexes.items():
        for zero_based_index, row in enumerate(rows):
            if row["label"] != "background":
                continue
            capture_index = zero_based_index + 1
            frame_distance, time_distance_ms = relevant_distance(rows, zero_based_index)
            boundary = frame_distance <= 3
            pair = pairs.get((passage_id, capture_index))
            temporal_valid = zero_based_index > 0 and rows[zero_based_index - 1]["label"] != "ruido"
            high_pdi = bool(
                temporal_valid
                and pair is not None
                and float(pair[PDI_FEATURE]) >= PDI_THRESHOLD
                and time_distance_ms <= 1000.0
            )
            if not boundary and not high_pdi:
                continue
            reason = (
                "boundary+high_pdi"
                if boundary and high_pdi
                else "boundary"
                if boundary
                else "high_pdi"
            )
            output.append(
                {
                    "passage_id": passage_id,
                    "capture_index": capture_index,
                    "relative_time_ms": float(row["relative_time_ms"]),
                    "rgb_filename": row["rgb_filename"],
                    "original_label": row["label"],
                    "candidate_reason": reason,
                }
            )
    output.sort(key=lambda item: (item["passage_id"], int(item["capture_index"])))
    return output


def assert_expected_scope(candidates: list[dict]) -> None:
    keys = [(row["passage_id"], int(row["capture_index"])) for row in candidates]
    if len(keys) != len(set(keys)):
        raise AssertionError("candidate queue contains duplicates")
    if any(row["original_label"] != "background" for row in candidates):
        raise AssertionError("non-background frame entered relabeling queue")
    counts = Counter(row["candidate_reason"] for row in candidates)
    if len(candidates) != EXPECTED_CANDIDATES or dict(counts) != EXPECTED_SCOPE:
        raise AssertionError(
            f"directed scope changed: total={len(candidates)} reasons={dict(counts)}"
        )
    passages = {row["passage_id"] for row in candidates}
    if len(passages) != EXPECTED_COHORT[0]:
        raise AssertionError(f"candidate passage count changed: {len(passages)}")


def neighbor_value(rows: list[dict], target_index: int, offset: int) -> str:
    position = target_index + offset
    if position < 0 or position >= len(rows):
        return ""
    value = rows[position].get("rgb_filename")
    return "" if value is None else str(value)


def build_manifest(
    candidates: list[dict], indexes: dict[str, list[dict]]
) -> list[dict]:
    """Cria a fila RGB autoritativa ordenada por passagem e capture index."""

    output = []
    for review_order, candidate in enumerate(candidates, start=1):
        row = dict(candidate)
        passage_rows = indexes[row["passage_id"]]
        target = int(row["capture_index"]) - 1
        for offset, field in (
            (-3, "rgb_prev_3"),
            (-2, "rgb_prev_2"),
            (-1, "rgb_prev_1"),
            (1, "rgb_next_1"),
            (2, "rgb_next_2"),
            (3, "rgb_next_3"),
        ):
            row[field] = neighbor_value(passage_rows, target, offset)
        row.update(
            {
                "review_order": review_order,
                "final_review": "",
                "notes": "",
            }
        )
        output.append(row)
    return output


def distributed_sample(
    rows: list[dict],
    count: int,
    rng: random.Random,
    passages_already_selected: set[str] | None = None,
) -> list[dict]:
    """Amostra uma linha por passagem antes de repetir uma passagem."""

    passages_already_selected = passages_already_selected or set()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["passage_id"]].append(row)
    for values in grouped.values():
        rng.shuffle(values)
    passages = list(grouped)
    rng.shuffle(passages)
    passages.sort(key=lambda passage_id: passage_id in passages_already_selected)
    output = []
    round_index = 0
    while len(output) < min(count, len(rows)):
        emitted = False
        for passage_id in passages:
            values = grouped[passage_id]
            if round_index < len(values):
                output.append(values[round_index])
                emitted = True
                if len(output) == min(count, len(rows)):
                    break
        if not emitted:
            break
        round_index += 1
    return output


def build_pilot(manifest: list[dict]) -> list[dict]:
    rng = random.Random(PILOT_SEED)
    selected = []
    represented_passages: set[str] = set()
    for reason in ("boundary+high_pdi", "high_pdi", "boundary"):
        stratum = [row for row in manifest if row["candidate_reason"] == reason]
        sample = distributed_sample(
            stratum,
            PILOT_PER_REASON,
            rng,
            passages_already_selected=represented_passages,
        )
        selected.extend(sample)
        represented_passages.update(row["passage_id"] for row in sample)
    selected.sort(key=lambda row: (row["passage_id"], int(row["capture_index"])))
    return [
        {"pilot_order": pilot_order, **row}
        for pilot_order, row in enumerate(selected, start=1)
    ]


def build_passage_summary(manifest: list[dict]) -> list[dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in manifest:
        grouped[row["passage_id"]].append(int(row["capture_index"]))
    return [
        {
            "passage_id": passage_id,
            "candidate_count": len(indexes),
            "min_capture_index": min(indexes),
            "max_capture_index": max(indexes),
        }
        for passage_id, indexes in sorted(grouped.items())
    ]


def validate_manifest(
    manifest: list[dict],
    indexes: dict[str, list[dict]],
    cohort_passages: set[str],
    require_complete: bool = False,
) -> None:
    if len(manifest) != EXPECTED_CANDIDATES:
        raise ValueError(f"manifest candidate count changed: {len(manifest)}")
    keys = []
    previous_key: tuple[str, int] | None = None
    for row in manifest:
        passage_id = row["passage_id"]
        capture_index = int(row["capture_index"])
        key = (passage_id, capture_index)
        if previous_key is not None and key < previous_key:
            raise ValueError("main RGB queue is not ordered by passage_id/capture_index")
        previous_key = key
        if passage_id not in cohort_passages:
            raise ValueError(f"passage outside operational cohort: {passage_id}")
        source = indexes[passage_id][capture_index - 1]
        if source["label"] != "background" or row["original_label"] != "background":
            raise ValueError(f"non-background manifest row: {passage_id} #{capture_index}")
        if row["rgb_filename"] != source["rgb_filename"]:
            raise ValueError(f"rgb filename mismatch: {passage_id} #{capture_index}")
        target = capture_index - 1
        for offset, field in (
            (-3, "rgb_prev_3"),
            (-2, "rgb_prev_2"),
            (-1, "rgb_prev_1"),
            (1, "rgb_next_1"),
            (2, "rgb_next_2"),
            (3, "rgb_next_3"),
        ):
            expected = neighbor_value(indexes[passage_id], target, offset)
            if row[field] != expected:
                raise ValueError(f"RGB neighbor mismatch: {passage_id} #{capture_index} {field}")
        final_review = row["final_review"].strip()
        if final_review not in FINAL_REVIEW_VALUES:
            raise ValueError(f"invalid final_review {final_review!r}: {passage_id} #{capture_index}")
        if require_complete and not final_review:
            raise ValueError(f"missing RGB review: {passage_id} #{capture_index}")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate candidate in manifest")


def assert_indexes_unchanged(data_root: Path, hashes_before: dict[str, str]) -> None:
    paths = index_paths(data_root, hashes_before)
    changed = [
        passage_id
        for passage_id, before in hashes_before.items()
        if file_sha256(paths[passage_id]) != before
    ]
    if changed:
        raise AssertionError(f"simulation indexes changed: {changed[:10]}")


def consolidate_relabels(manifest: list[dict]) -> list[dict]:
    output = []
    for row in manifest:
        final_review = row["final_review"].strip()
        if not final_review:
            continue
        if final_review not in FINAL_TO_TARGET:
            raise ValueError(f"cannot consolidate final label: {final_review!r}")
        output.append(
            {
                **row,
                "visual_event_label": FINAL_TO_TARGET[final_review],
                "training_eligibility": "NO" if final_review == "AMBIGUOUS" else "YES",
            }
        )
    output.sort(key=lambda row: (row["passage_id"], int(row["capture_index"])))
    return output


def build_command(args) -> None:
    manifest_path = args.workflow_dir / "review_manifest.csv"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"{manifest_path} already exists; use --force only before manual review"
        )
    passage_ids, indexes, hashes_before = load_operational_indexes(
        args.data_root, args.cohort_metrics
    )
    pairs = label_audit.pair_lookup(args.pair_features)
    candidates = build_candidates(indexes, pairs)
    assert_expected_scope(candidates)
    manifest = build_manifest(candidates, indexes)
    validate_manifest(manifest, indexes, set(passage_ids))
    pilot = build_pilot(manifest)
    passage_summary = build_passage_summary(manifest)
    write_csv(manifest_path, manifest, MANIFEST_FIELDS)
    write_csv(args.workflow_dir / "pilot_manifest.csv", pilot, ["pilot_order", *MANIFEST_FIELDS])
    write_csv(
        args.workflow_dir / "passage_summary.csv",
        passage_summary,
        PASSAGE_SUMMARY_FIELDS,
    )
    write_csv(args.workflow_dir / "relabels.csv", [], RELABEL_FIELDS)
    assert_indexes_unchanged(args.data_root, hashes_before)
    summary = {
        "candidates": len(manifest),
        "candidate_reasons": dict(Counter(row["candidate_reason"] for row in manifest)),
        "passages": len(passage_summary),
        "pilot_candidates": len(pilot),
        "pilot_reasons": dict(Counter(row["candidate_reason"] for row in pilot)),
        "pilot_passages": len({row["passage_id"] for row in pilot}),
        "review_source": "RGB files consulted manually in Google Drive",
        "rgb_access_by_script": "metadata only; no RGB file access",
        "depth_panels_required": False,
        "pdi_feature_reused": PDI_FEATURE,
        "pdi_threshold_reused": PDI_THRESHOLD,
    }
    (args.workflow_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def load_and_validate(args, require_complete: bool = False):
    passage_ids, indexes, hashes_before = load_operational_indexes(
        args.data_root, args.cohort_metrics
    )
    manifest = read_csv(args.manifest)
    validate_manifest(manifest, indexes, set(passage_ids), require_complete=require_complete)
    assert_indexes_unchanged(args.data_root, hashes_before)
    return manifest, indexes


def consolidate_command(args) -> None:
    manifest, _ = load_and_validate(args, require_complete=args.require_complete)
    relabels = consolidate_relabels(manifest)
    write_csv(args.output, relabels, RELABEL_FIELDS)
    print(f"Consolidated relabels: {len(relabels)} rows -> {args.output}")


def validate_command(args) -> None:
    manifest, _ = load_and_validate(args, require_complete=args.require_complete)
    print(
        json.dumps(
            {
                "valid": True,
                "rows": len(manifest),
                "reasons": dict(Counter(row["candidate_reason"] for row in manifest)),
                "final_review": dict(Counter(row["final_review"] for row in manifest)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def common_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=WORKFLOW_DIR / "review_manifest.csv")
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--cohort-metrics", type=Path, default=base.DEFAULT_COHORT_METRICS)
    parser.add_argument("--require-complete", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build main RGB queue and pilot")
    build.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    build.add_argument("--cohort-metrics", type=Path, default=base.DEFAULT_COHORT_METRICS)
    build.add_argument("--pair-features", type=Path, default=DEFAULT_PAIR_FEATURES)
    build.add_argument("--workflow-dir", type=Path, default=WORKFLOW_DIR)
    build.add_argument("--force", action="store_true")
    build.set_defaults(function=build_command)

    consolidate = subparsers.add_parser("consolidate", help="build final relabels")
    common_manifest_arguments(consolidate)
    consolidate.add_argument("--output", type=Path, default=WORKFLOW_DIR / "relabels.csv")
    consolidate.set_defaults(function=consolidate_command)

    validate = subparsers.add_parser("validate", help="validate RGB review manifest")
    common_manifest_arguments(validate)
    validate.set_defaults(function=validate_command)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
