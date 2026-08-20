#!/usr/bin/env python3
"""Avaliação da Arquitetura Acoplada (Visual + Pipeline Acoplados em IDLE)
Comparando LOW = 3 FPS, LOW = 4 FPS e LOW = 5 FPS com Selection Hold N=2.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ANALYSIS = REPO_ROOT / "data-analysis"
for path in (
    REPO_ROOT,
    DATA_ANALYSIS,
    DATA_ANALYSIS / "visual_event_quality_gate_audit",
    DATA_ANALYSIS / "visual_event_preprocessing_ablation",
    DATA_ANALYSIS / "selection_hold_evaluation",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import visual_event_diagnostic as base
from audit_quality_gate import (
    FEATURES_CSV,
    read_rows,
)
from review_and_rerun_baseline import (
    existing_score_lookup,
)
from eval_selection_hold import (
    get_visual_post_states,
    load_materialized_selection_decisions,
    simulate_adaptive_passage,
)
from eval_visual_frequency_diagnosis import (
    aggregate_cohort,
)

OUTPUT_DIR = DATA_ANALYSIS / "selection_hold_evaluation" / "output"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    passage_ids = base.load_cohort(base.DEFAULT_COHORT_METRICS)
    indexes = base.load_indexes(base.DEFAULT_DATA_ROOT, passage_ids)
    feature_rows = read_rows(FEATURES_CSV)
    features = {(row["passage_id"], int(row["capture_index"])): row for row in feature_rows}
    raw_scores = existing_score_lookup()

    visual_states, threshold, direction = get_visual_post_states(indexes, features, raw_scores)
    selection_decisions = load_materialized_selection_decisions()

    low_rates = [2.0, 3.0, 4.0, 5.0]
    n_hold = 2

    summaries = {}
    passage_details_by_fps = {}

    for low_fps in low_rates:
        fps_str = f"{int(low_fps)}fps" if low_fps.is_integer() else f"{low_fps}fps"
        passage_results = [
            simulate_adaptive_passage(
                passage_id=tag,
                frames=indexes[tag],
                visual_active_series=visual_states[tag],
                selection_decisions=selection_decisions[tag],
                n_hold=n_hold,
                low_fps=low_fps,
            )
            for tag in passage_ids
        ]
        passage_details_by_fps[low_fps] = passage_results
        agg = aggregate_cohort(passage_results, n_hold=n_hold)
        agg["low_fps"] = low_fps
        agg["reduction_vs_baseline_pct"] = (1.0 - (agg["total_frames_captured"] / 13741.0)) * 100.0
        summaries[low_fps] = agg

        # Salvar CSV de passagens para este FPS
        csv_filename = OUTPUT_DIR / f"coupled_n2_low_{fps_str}_by_passage.csv"
        with csv_filename.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "passage_id",
                    "n_frames_total",
                    "n_frames_captured",
                    "n_suited_available",
                    "n_suited_captured",
                    "suited_passage_covered",
                    "suited_retention",
                    "lost_suited_opportunities",
                    "lost_suited_passage",
                    "n_accepted_captured",
                    "n_rejected_captured",
                    "time_low_ms",
                    "time_high_ms",
                    "total_time_ms",
                    "pct_time_low",
                    "pct_time_high",
                    "transitions_low_to_high",
                    "transitions_high_to_low",
                    "hold_prevented_downshift_count",
                    "hold_recovered_suited_frames",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(passage_results)

    # Identificar passagens perdidas para cada FPS
    lost_passages = {}
    for low_fps, results in passage_details_by_fps.items():
        lost = [r["passage_id"] for r in results if r["lost_suited_passage"]]
        lost_passages[low_fps] = lost

    # Salvar Resumo Consolidado
    summary_file = OUTPUT_DIR / "coupled_sweep_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summaries": summaries,
                "lost_passages": lost_passages,
            },
            f,
            indent=2,
        )

    print("=== RESUMO CONSOLIDADO ACOPLADO (N=2) ===")
    for low_fps in [2.0, 3.0, 4.0, 5.0]:
        s = summaries[low_fps]
        print(f"\n--- LOW = {low_fps} FPS ---")
        print(f"Coverage: {s['covered_suited_passages']}/{s['passages_with_suited']} ({s['suited_passage_coverage_pct']:.2f}%)")
        print(f"Passagens perdidas: {lost_passages[low_fps]}")
        print(f"Retention Suited: {s['total_suited_captured']}/{s['total_suited_available']} ({s['suited_frame_retention_pct']:.2f}%)")
        print(f"Suited perdidos: {s['lost_suited_opportunities']}")
        print(f"Total Frames: {s['total_frames_captured']} (Redução vs 13.741: {s['reduction_vs_baseline_pct']:.2f}%)")
        print(f"Tempo LOW: {s['total_time_low_s']:.1f}s ({s['pct_time_low']:.2f}%) | Tempo HIGH: {s['total_time_high_s']:.1f}s ({s['pct_time_high']:.2f}%)")
        print(f"Transições L->H: {s['transitions_low_to_high']} | H->L: {s['transitions_high_to_low']}")
        print(f"Accepted: {s['total_accepted_captured']} | Rejected: {s['total_rejected_captured']}")
        print(f"Hold vetos: {s['hold_prevented_downshift_count']} | Suited recuperados pelo Hold: {s['hold_recovered_suited_frames']}")
        print(f"Duração Episódios HIGH (ms): Média={s['mean_high_episode_duration_ms']:.1f}, Mediana={s['median_high_episode_duration_ms']:.1f}, P95={s['p95_high_episode_duration_ms']:.1f}")


if __name__ == "__main__":
    main()
