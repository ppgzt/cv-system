#!/usr/bin/env python3
"""Reconstrói offline as oportunidades de captura e sua retenção pelo selector.

O script não carrega modelos nem depende de Raspberry Pi. O schedule Fixed-FPS
é o mesmo usado por ``ThreadPipeline`` e os labels vêm diretamente de
``simulation_index.json``. Dados observados do selector são opcionais.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domain.helpers.capture_schedule import build_fixed_fps_schedule  # noqa: E402


DEFAULT_FPS = (1, 2, 3, 4, 5, 10, 15, 20, 30)
DEFAULT_COHORT_METRICS = (
    "power_runs/battery_mas-single_20260708_104924/"
    "mas-single_1fps_r1/**/metrics.json"
)
DEFAULT_SELECTOR_METRICS = (
    "power_runs/battery_mas-single_20260708_104924/"
    "mas-single_{fps:g}fps_r1/**/metrics.json"
)
ALLOWED_LABELS = {"suited", "background", "parcial", "ruido"}
ARTICLE_COHORT_EXPECTED = {
    "n_evaluated_passages": 184,
    "n_evaluated_frames": 13_741,
    "n_evaluated_suited_frames": 1_655,
}
ARTICLE_GOLDEN_ZERO_INFERENCE = {
    1.0: (72, 3),
    2.0: (27, 2),
    3.0: (8, 1),
    4.0: (1, 1),
}

BY_PASSAGE_HEADER = [
    "condition",
    "fps",
    "passage_id",
    "n_capture_events",
    "n_unique_source_frames",
    "n_human_suited_capture_events",
    "n_available_human_suited_source_frames",
    "n_captured_unique_human_suited_source_frames",
    "unique_gt_suited_retention",
    "gt_opportunity_exists",
    "selector_data_available",
    "selector_event_mapping_available",
    "n_classifier_accepted_events",
    "classifier_covered",
    "n_human_suited_events_preserved",
    "gt_suited_opportunity_preserved",
    "coverage_preserved_by_false_positive",
    "coverage_outcome",
]

SUMMARY_HEADER = [
    "condition",
    "fps",
    "n_passages",
    "n_gt_covered",
    "gt_coverage",
    "n_gt_uncovered",
    "n_sampling_failures",
    "n_capture_events",
    "n_human_suited_capture_events",
    "human_suited_capture_event_ratio",
    "n_unique_source_frames",
    "n_available_human_suited_source_frames",
    "n_captured_unique_human_suited_source_frames",
    "unique_gt_suited_retention",
    "selector_data_available",
    "selector_event_mapping_available",
    "n_classifier_covered",
    "classifier_coverage",
    "n_zero_inference_passages",
    "n_coverage_preserved",
    "coverage_preserved_proportion",
    "n_selector_side_coverage_losses",
    "selector_side_coverage_loss_proportion",
    "selector_side_loss_given_gt_opportunity",
    "n_gt_suited_opportunity_preserved",
    "gt_suited_opportunity_preservation_rate",
    "n_coverage_preserved_by_false_positive",
]


@dataclass(frozen=True)
class DatasetAudit:
    n_dataset_passages: int
    n_dataset_frames: int
    n_evaluated_passages: int
    n_evaluated_frames: int
    n_evaluated_suited_frames: int
    label_counts: dict[str, int]


def validate_article_cohort(audit: DatasetAudit) -> None:
    """Falha cedo se o cohort padrão divergir dos números publicados."""
    for field, expected in ARTICLE_COHORT_EXPECTED.items():
        observed = getattr(audit, field)
        if observed != expected:
            raise ValueError(
                f"cohort operacional divergente: {field}={observed}, "
                f"esperado={expected}"
            )


def validate_article_golden(summaries: list[dict]) -> None:
    """Confere a decomposição independente dos casos sem inferência."""
    by_fps = {float(row["fps"]): row for row in summaries}
    for fps, (expected_sampling, expected_selector) in (
        ARTICLE_GOLDEN_ZERO_INFERENCE.items()
    ):
        row = by_fps.get(fps)
        if row is None:
            raise ValueError(f"golden validation sem resultado para {fps:g} FPS")
        observed = (
            row["n_sampling_failures"],
            row["n_selector_side_coverage_losses"],
        )
        expected = (expected_sampling, expected_selector)
        if observed != expected:
            raise ValueError(
                f"golden validation divergiu em {fps:g} FPS: "
                f"observado={observed}, esperado={expected}"
            )


def _repo_relative(path_or_pattern: str) -> str:
    path = Path(path_or_pattern)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def resolve_one_file(path_or_pattern: str, description: str) -> Path:
    matches = sorted(glob.glob(_repo_relative(path_or_pattern), recursive=True))
    files = [Path(match) for match in matches if Path(match).is_file()]
    if len(files) != 1:
        raise ValueError(
            f"{description}: esperado exatamente um arquivo para "
            f"{path_or_pattern!r}, encontrados {len(files)}"
        )
    return files[0]


def load_metrics(path_or_pattern: str, description: str) -> dict:
    path = resolve_one_file(path_or_pattern, description)
    with path.open() as file:
        metrics = json.load(file)
    if not isinstance(metrics.get("animals"), dict):
        raise ValueError(f"{description}: campo 'animals' ausente ou inválido em {path}")
    return metrics


def load_cohort(path_or_pattern: str) -> list[str]:
    metrics = load_metrics(path_or_pattern, "métricas do cohort")
    return sorted(metrics["animals"])


def audit_and_load_dataset(
    data_root: Path,
    passage_ids: Iterable[str] | None = None,
) -> tuple[dict[str, list[dict]], DatasetAudit]:
    tags_root = data_root / "animal-tags"
    depth_root = data_root / "DEPTH"
    dataset_tags = sorted(
        path.name
        for path in tags_root.iterdir()
        if path.is_dir() and (path / "simulation_index.json").is_file()
    )
    selected_tags = dataset_tags if passage_ids is None else sorted(passage_ids)
    missing_tags = sorted(set(selected_tags) - set(dataset_tags))
    if missing_tags:
        raise ValueError(f"passagens do cohort ausentes no dataset: {missing_tags}")

    indexes: dict[str, list[dict]] = {}
    label_counts = {label: 0 for label in sorted(ALLOWED_LABELS)}
    all_source_keys: dict[tuple[str, str], tuple[float, str]] = {}
    n_dataset_frames = 0
    n_evaluated_frames = 0
    n_evaluated_suited_frames = 0

    for tag in dataset_tags:
        index_path = tags_root / tag / "simulation_index.json"
        with index_path.open() as file:
            index = json.load(file)
        if not isinstance(index, list) or not index:
            raise ValueError(f"índice vazio ou inválido: {index_path}")

        seen_times: set[float] = set()
        seen_names: set[str] = set()
        indexed_names: set[str] = set()
        for position, frame in enumerate(index):
            missing_fields = {
                field
                for field in ("relative_time_ms", "depth_filename", "label")
                if field not in frame or frame[field] is None
            }
            if missing_fields:
                raise ValueError(
                    f"{tag}[{position}] sem campos obrigatórios: "
                    f"{sorted(missing_fields)}"
                )
            timestamp = float(frame["relative_time_ms"])
            filename = str(frame["depth_filename"])
            label = str(frame["label"])
            if label not in ALLOWED_LABELS:
                raise ValueError(f"label desconhecido em {tag}/{filename}: {label!r}")
            if timestamp in seen_times:
                raise ValueError(f"timestamp duplicado em {tag}: {timestamp}")
            if filename in seen_names:
                raise ValueError(f"filename duplicado em {tag}: {filename}")
            source_key = (tag, filename)
            previous = all_source_keys.get(source_key)
            identity = (timestamp, label)
            if previous is not None and previous != identity:
                raise ValueError(
                    "(passage_id, filename) não identifica inequivocamente "
                    f"um frame: {source_key}"
                )
            if not (depth_root / tag / filename).is_file():
                raise ValueError(f"depth frame ausente: {depth_root / tag / filename}")
            seen_times.add(timestamp)
            seen_names.add(filename)
            indexed_names.add(filename)
            all_source_keys[source_key] = identity

        stored_names = {
            path.name for path in (depth_root / tag).iterdir() if path.is_file()
        }
        unindexed = sorted(stored_names - indexed_names)
        if unindexed:
            raise ValueError(
                f"frames armazenados sem entrada no índice em {tag}: {unindexed[:5]}"
            )

        index.sort(key=lambda frame: frame["relative_time_ms"])
        n_dataset_frames += len(index)
        if tag in selected_tags:
            indexes[tag] = index
            n_evaluated_frames += len(index)
            for frame in index:
                label_counts[frame["label"]] += 1
                if frame["label"] == "suited":
                    n_evaluated_suited_frames += 1

    return indexes, DatasetAudit(
        n_dataset_passages=len(dataset_tags),
        n_dataset_frames=n_dataset_frames,
        n_evaluated_passages=len(selected_tags),
        n_evaluated_frames=n_evaluated_frames,
        n_evaluated_suited_frames=n_evaluated_suited_frames,
        label_counts=label_counts,
    )


def load_selector_observations(
    template: str | None,
    fps: float,
    passage_ids: Iterable[str],
) -> tuple[dict[str, dict], bool] | None:
    if template is None:
        return None
    pattern = template.format(fps=fps)
    metrics = load_metrics(pattern, f"métricas observadas do selector a {fps:g} FPS")
    animals = metrics["animals"]
    expected = set(passage_ids)
    if set(animals) != expected:
        missing = sorted(expected - set(animals))
        extra = sorted(set(animals) - expected)
        raise ValueError(
            f"cohort divergente nas métricas de {fps:g} FPS; "
            f"ausentes={missing}, extras={extra}"
        )
    nonempty_keys = [
        str(key)
        for observation in animals.values()
        for key in observation.get("imgs", {})
    ]
    numeric_keys = [key.isdigit() for key in nonempty_keys]
    if any(numeric_keys) and not all(numeric_keys):
        raise ValueError(
            f"schema misto de imgs.keys() nas métricas de {fps:g} FPS"
        )
    # No ThreadPipeline atual as chaves são capture_index. Outros engines
    # podem persistir UUID/frame_id; passage coverage continua disponível,
    # mas a retenção evento-a-evento fica deliberadamente indisponível.
    event_mapping_available = bool(nonempty_keys) and all(numeric_keys)
    return animals, event_mapping_available


def analyze_oracle_coverage(
    indexes: dict[str, list[dict]],
    fps_values: Iterable[float],
    selector_metrics_template: str | None = None,
    max_passage_seconds: float | None = None,
) -> tuple[list[dict], list[dict]]:
    if max_passage_seconds is not None and max_passage_seconds < 0:
        raise ValueError("max_passage_seconds must not be negative")
    by_passage: list[dict] = []
    summaries: list[dict] = []
    passage_ids = sorted(indexes)

    for fps in fps_values:
        fps = float(fps)
        selector_result = load_selector_observations(
            selector_metrics_template, fps, passage_ids
        )
        selector = selector_result[0] if selector_result is not None else None
        selector_event_mapping_available = (
            selector_result[1] if selector_result is not None else False
        )
        fps_rows = []

        for tag in passage_ids:
            frames = indexes[tag]
            times = np.array(
                [frame["relative_time_ms"] for frame in frames], dtype=float
            )
            end_ms = None
            if max_passage_seconds is not None:
                end_ms = min(
                    float(times[-1]),
                    float(times[0]) + max_passage_seconds * 1000.0,
                )
            schedule = build_fixed_fps_schedule(times, fps, end_ms=end_ms)
            selected_frames = [frames[event.source_index] for event in schedule]
            suited_event_indices = {
                event_index
                for event_index, frame in enumerate(selected_frames, start=1)
                if frame["label"] == "suited"
            }
            unique_sources = {frame["depth_filename"] for frame in selected_frames}
            unique_suited_sources = {
                frame["depth_filename"]
                for frame in selected_frames
                if frame["label"] == "suited"
            }
            available_suited_sources = {
                frame["depth_filename"]
                for frame in frames
                if frame["label"] == "suited"
            }

            selector_available = selector is not None
            accepted_indices: set[int] = set()
            classifier_covered = None
            preserved_count = None
            gt_suited_preserved = None
            if selector_available:
                observation = selector[tag]
                observed_total = int(observation["total_of_images"])
                if observed_total != len(schedule):
                    raise ValueError(
                        f"count de captura divergente em {fps:g} FPS/{tag}: "
                        f"reconstruído={len(schedule)}, observado={observed_total}"
                    )
                suitable_images = int(observation.get("suitable_images", 0))
                observed_images = observation.get("imgs", {})
                if suitable_images != len(observed_images):
                    raise ValueError(
                        f"suitable_images diverge de imgs em {fps:g} FPS/{tag}: "
                        f"{suitable_images} != {len(observed_images)}"
                    )
                classifier_covered = suitable_images > 0
                if selector_event_mapping_available:
                    accepted_indices = {int(key) for key in observed_images}
                    invalid_indices = sorted(
                        index for index in accepted_indices
                        if index < 1 or index > len(schedule)
                    )
                    if invalid_indices:
                        raise ValueError(
                            "índices aceitos fora do schedule em "
                            f"{fps:g} FPS/{tag}: {invalid_indices}"
                        )
                    preserved_count = len(accepted_indices & suited_event_indices)
                    gt_suited_preserved = preserved_count > 0
                elif not classifier_covered:
                    # Sem nenhuma aceitação, sabemos que nenhuma oportunidade
                    # GT foi preservada mesmo sem um mapping evento-a-evento.
                    preserved_count = 0
                    gt_suited_preserved = False

            gt_exists = bool(suited_event_indices)
            if not selector_available:
                coverage_outcome = None
            elif classifier_covered:
                coverage_outcome = "coverage_preserved"
            elif not gt_exists:
                coverage_outcome = "sampling_failure"
            else:
                coverage_outcome = "selector_side_coverage_loss"

            coverage_preserved_by_false_positive = None
            if classifier_covered and gt_suited_preserved is not None:
                coverage_preserved_by_false_positive = not gt_suited_preserved

            unique_gt_retention = (
                len(unique_suited_sources) / len(available_suited_sources)
                if available_suited_sources else None
            )

            row = {
                "condition": "fixed_fps",
                "fps": fps,
                "passage_id": tag,
                "n_capture_events": len(schedule),
                "n_unique_source_frames": len(unique_sources),
                "n_human_suited_capture_events": len(suited_event_indices),
                "n_available_human_suited_source_frames": (
                    len(available_suited_sources)
                ),
                "n_captured_unique_human_suited_source_frames": (
                    len(unique_suited_sources)
                ),
                "unique_gt_suited_retention": unique_gt_retention,
                "gt_opportunity_exists": gt_exists,
                "selector_data_available": selector_available,
                "selector_event_mapping_available": (
                    selector_event_mapping_available
                ),
                "n_classifier_accepted_events": (
                    int(selector[tag].get("suitable_images", 0))
                    if selector_available else None
                ),
                "classifier_covered": classifier_covered,
                "n_human_suited_events_preserved": preserved_count,
                "gt_suited_opportunity_preserved": gt_suited_preserved,
                "coverage_preserved_by_false_positive": (
                    coverage_preserved_by_false_positive
                ),
                "coverage_outcome": coverage_outcome,
            }
            fps_rows.append(row)
            by_passage.append(row)

        n_passages = len(fps_rows)
        n_gt_covered = sum(row["gt_opportunity_exists"] for row in fps_rows)
        n_capture_events = sum(row["n_capture_events"] for row in fps_rows)
        n_suited_events = sum(
            row["n_human_suited_capture_events"] for row in fps_rows
        )
        selector_available = selector is not None
        n_classifier_covered = (
            sum(bool(row["classifier_covered"]) for row in fps_rows)
            if selector_available else None
        )
        n_coverage_preserved = (
            sum(row["coverage_outcome"] == "coverage_preserved" for row in fps_rows)
            if selector_available else None
        )
        n_sampling_failures = (
            sum(row["coverage_outcome"] == "sampling_failure" for row in fps_rows)
            if selector_available else None
        )
        n_selector_losses = (
            sum(
                row["coverage_outcome"] == "selector_side_coverage_loss"
                for row in fps_rows
            ) if selector_available else None
        )
        available_suited_sources = sum(
            row["n_available_human_suited_source_frames"] for row in fps_rows
        )
        captured_unique_suited_sources = sum(
            row["n_captured_unique_human_suited_source_frames"]
            for row in fps_rows
        )
        n_gt_suited_preserved = (
            sum(bool(row["gt_suited_opportunity_preserved"]) for row in fps_rows)
            if selector_event_mapping_available else None
        )
        n_false_positive_coverage = (
            sum(bool(row["coverage_preserved_by_false_positive"]) for row in fps_rows)
            if selector_event_mapping_available else None
        )
        summaries.append({
            "condition": "fixed_fps",
            "fps": fps,
            "n_passages": n_passages,
            "n_gt_covered": n_gt_covered,
            "gt_coverage": n_gt_covered / n_passages,
            "n_gt_uncovered": n_passages - n_gt_covered,
            "n_sampling_failures": n_sampling_failures,
            "n_capture_events": n_capture_events,
            "n_human_suited_capture_events": n_suited_events,
            "human_suited_capture_event_ratio": n_suited_events / n_capture_events,
            "n_unique_source_frames": sum(
                row["n_unique_source_frames"] for row in fps_rows
            ),
            "n_available_human_suited_source_frames": available_suited_sources,
            "n_captured_unique_human_suited_source_frames": (
                captured_unique_suited_sources
            ),
            "unique_gt_suited_retention": (
                captured_unique_suited_sources / available_suited_sources
                if available_suited_sources else None
            ),
            "selector_data_available": selector_available,
            "selector_event_mapping_available": (
                selector_event_mapping_available
            ),
            "n_classifier_covered": n_classifier_covered,
            "classifier_coverage": (
                n_classifier_covered / n_passages
                if selector_available else None
            ),
            "n_zero_inference_passages": (
                n_passages - n_classifier_covered
                if selector_available else None
            ),
            "n_coverage_preserved": n_coverage_preserved,
            "coverage_preserved_proportion": (
                n_coverage_preserved / n_passages
                if selector_available else None
            ),
            "n_selector_side_coverage_losses": n_selector_losses,
            "selector_side_coverage_loss_proportion": (
                n_selector_losses / n_passages if selector_available else None
            ),
            "selector_side_loss_given_gt_opportunity": (
                n_selector_losses / n_gt_covered
                if selector_available and n_gt_covered else None
            ),
            "n_gt_suited_opportunity_preserved": n_gt_suited_preserved,
            "gt_suited_opportunity_preservation_rate": (
                n_gt_suited_preserved / n_gt_covered
                if n_gt_suited_preserved is not None and n_gt_covered else None
            ),
            "n_coverage_preserved_by_false_positive": n_false_positive_coverage,
        })

    return summaries, by_passage


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/exp1")
    parser.add_argument("--fps", nargs="+", type=float, default=DEFAULT_FPS)
    parser.add_argument("--cohort-metrics", default=DEFAULT_COHORT_METRICS)
    parser.add_argument(
        "--all-passages",
        action="store_true",
        help="avalia todas as passagens atuais em vez do cohort das métricas",
    )
    parser.add_argument(
        "--selector-metrics-template",
        default=DEFAULT_SELECTOR_METRICS,
        help="template com {fps}; use 'none' para análise somente GT",
    )
    parser.add_argument(
        "--output-dir", default="data-analysis/oracle_coverage_output"
    )
    parser.add_argument(
        "--max-passage-seconds",
        type=float,
        default=None,
        help="cap temporal opcional, com a mesma semântica do ThreadPipeline",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = Path(_repo_relative(args.data_root))
    passage_ids = None if args.all_passages else load_cohort(args.cohort_metrics)
    indexes, audit = audit_and_load_dataset(data_root, passage_ids)
    selector_template = args.selector_metrics_template
    if selector_template.lower() in {"none", "null", ""}:
        selector_template = None

    standard_article_mode = (
        not args.all_passages
        and args.data_root == "data/exp1"
        and args.cohort_metrics == DEFAULT_COHORT_METRICS
    )
    if standard_article_mode:
        validate_article_cohort(audit)

    summaries, by_passage = analyze_oracle_coverage(
        indexes,
        args.fps,
        selector_template,
        max_passage_seconds=args.max_passage_seconds,
    )
    standard_golden_mode = (
        standard_article_mode
        and selector_template == DEFAULT_SELECTOR_METRICS
        and tuple(args.fps) == tuple(float(fps) for fps in DEFAULT_FPS)
        and args.max_passage_seconds is None
    )
    if standard_golden_mode:
        validate_article_golden(summaries)
    output_dir = Path(_repo_relative(args.output_dir))
    summary_path = output_dir / "oracle_coverage_summary.csv"
    passage_path = output_dir / "oracle_coverage_by_passage.csv"
    write_csv(summary_path, SUMMARY_HEADER, summaries)
    write_csv(passage_path, BY_PASSAGE_HEADER, by_passage)

    print(
        "Dataset audit: "
        f"{audit.n_dataset_passages} passagens, {audit.n_dataset_frames} frames; "
        f"cohort avaliado: {audit.n_evaluated_passages} passagens, "
        f"{audit.n_evaluated_frames} frames, "
        f"{audit.n_evaluated_suited_frames} suited."
    )
    print(f"Labels no cohort: {audit.label_counts}")
    print(f"Resumo: {summary_path}")
    print(f"Por passagem: {passage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
