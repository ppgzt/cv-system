#!/usr/bin/env python3
"""Diagnóstico Detalhado da Queda de Suited Passage Coverage (184 -> 178) e Frequência do Visual.

Este script investiga:
1. Trace causal completo das 6 passagens perdidas ('0009vd', '0034az', '0060az', '0275', '0987', '0995');
2. Classificação objetiva da causa de cada perda;
3. Comparação com a timeline de ativação em Original-Timing;
4. Teste diagnóstico: Arquitetura A (Visual a 2 FPS) vs Arquitetura B (Visual Full-Rate Decoupled);
5. Teste diagnóstico de LOW: LOW = 2 FPS vs LOW = 3 FPS;
6. Análise de Custo-Benefício e Retenção do Selection Hold sob ambos os cenários.
"""

from __future__ import annotations

import csv
import json
import math
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
import run_ablation as ablation
from audit_quality_gate import (
    FEATURES_CSV,
    FRACTION_GE_2500_YOUDEN,
    P99_YOUDEN,
    read_rows,
)
from review_and_rerun_baseline import (
    build_baseline_series,
    existing_score_lookup,
)
from eval_selection_hold import (
    get_visual_post_states,
    load_materialized_selection_decisions,
)

OUTPUT_DIR = DATA_ANALYSIS / "selection_hold_evaluation" / "output"


def simulate_decoupled_visual_adaptive(
    passage_id: str,
    frames: list[dict],
    visual_active_series: list[bool],  # Visual observa full-rate trace
    selection_decisions: dict[int, bool],
    n_hold: int,
    low_fps: float = 2.0,
) -> dict[str, Any]:
    """Cenário B: Visual observa em Full-Rate, mas pipeline de admissão usa LOW/HIGH adaptativo."""
    n_frames = len(frames)
    timestamps = [float(f["relative_time_ms"]) for f in frames]
    labels = [f["label"] for f in frames]

    captured_indices: list[int] = []
    capture_rates: list[str] = []
    capture_times: list[float] = []

    current_rate = "LOW"
    hold_active = False
    consecutive_rejections = 0

    hold_prevented_downshift_count = 0
    hold_recovered_suited_frames = 0
    transitions_low_to_high = 0
    transitions_high_to_low = 0

    high_episodes_duration_ms: list[float] = []
    current_high_start_time: float | None = None

    next_low_scheduled_ms = timestamps[0]

    # No Cenário B, a cada frame nativo k (no instante t_k):
    # 1. Visual Event avalia frame k -> visual_active_series[k]
    # 2. Se visual_active_series[k] == True -> modo torna-se HIGH imediatamente para k e posteriores.
    # 3. Se modo é HIGH, frame k é admitido no pipeline.
    # 4. Se modo é LOW, frame k é admitido apenas se t_k >= next_low_scheduled_ms.
    for k in range(n_frames):
        t_current = timestamps[k]
        v_active = visual_active_series[k]

        # Determinar taxa desejada antes da admissão do frame k
        prev_rate = current_rate
        if v_active:
            target_rate = "HIGH"
        else:
            if current_rate == "HIGH" and hold_active and n_hold > 0:
                target_rate = "HIGH"
            else:
                target_rate = "LOW"
                hold_active = False
                consecutive_rejections = 0

        if prev_rate == "LOW" and target_rate == "HIGH":
            transitions_low_to_high += 1
            current_high_start_time = t_current
        elif prev_rate == "HIGH" and target_rate == "LOW":
            transitions_high_to_low += 1
            if current_high_start_time is not None:
                high_episodes_duration_ms.append(
                    t_current - current_high_start_time
                )
                current_high_start_time = None
            next_low_scheduled_ms = t_current + (1000.0 / low_fps)

        current_rate = target_rate

        # Decisão de admissão para o frame k
        admit = False
        if current_rate == "HIGH":
            admit = True
        else:
            if t_current >= next_low_scheduled_ms - 1e-5:
                admit = True
                next_low_scheduled_ms = t_current + (1000.0 / low_fps)

        if admit:
            captured_indices.append(k)
            capture_rates.append(current_rate)
            capture_times.append(t_current)

            # Selection avalia frame admitido
            s_accepted = selection_decisions[k + 1]
            if n_hold > 0 and current_rate == "HIGH":
                if s_accepted:
                    hold_active = True
                    consecutive_rejections = 0
                else:
                    if hold_active:
                        consecutive_rejections += 1
                        if consecutive_rejections >= n_hold:
                            hold_active = False

            if not v_active and current_rate == "HIGH" and hold_active:
                hold_prevented_downshift_count += 1
                if labels[k] == "suited":
                    hold_recovered_suited_frames += 1

    if current_rate == "HIGH" and current_high_start_time is not None:
        high_episodes_duration_ms.append(
            timestamps[-1] - current_high_start_time
        )

    total_passage_time_ms = (
        timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    )
    time_high_ms = sum(high_episodes_duration_ms)
    time_low_ms = max(0.0, total_passage_time_ms - time_high_ms)

    captured_set = set(captured_indices)
    suited_indices = [i for i, l in enumerate(labels) if l == "suited"]
    n_suited_available = len(suited_indices)
    captured_suited = [i for i in suited_indices if i in captured_set]
    n_suited_captured = len(captured_suited)

    accepted_captured = sum(selection_decisions[idx + 1] for idx in captured_indices)
    rejected_captured = len(captured_indices) - accepted_captured

    return {
        "passage_id": passage_id,
        "n_frames_total": n_frames,
        "n_frames_captured": len(captured_indices),
        "n_suited_available": n_suited_available,
        "n_suited_captured": n_suited_captured,
        "suited_passage_covered": (n_suited_captured > 0)
        if n_suited_available > 0
        else True,
        "suited_retention": (n_suited_captured / n_suited_available)
        if n_suited_available > 0
        else 1.0,
        "lost_suited_opportunities": n_suited_available - n_suited_captured,
        "lost_suited_passage": (n_suited_available > 0 and n_suited_captured == 0),
        "n_accepted_captured": accepted_captured,
        "n_rejected_captured": rejected_captured,
        "time_low_ms": time_low_ms,
        "time_high_ms": time_high_ms,
        "total_time_ms": total_passage_time_ms,
        "pct_time_low": (time_low_ms / total_passage_time_ms * 100.0)
        if total_passage_time_ms > 0
        else 0.0,
        "pct_time_high": (time_high_ms / total_passage_time_ms * 100.0)
        if total_passage_time_ms > 0
        else 0.0,
        "transitions_low_to_high": transitions_low_to_high,
        "transitions_high_to_low": transitions_high_to_low,
        "high_episodes_durations_ms": high_episodes_duration_ms,
        "hold_prevented_downshift_count": hold_prevented_downshift_count,
        "hold_recovered_suited_frames": hold_recovered_suited_frames,
        "captured_indices": captured_indices,
    }


def aggregate_cohort(passage_results: list[dict], n_hold: int) -> dict[str, Any]:
    total_passages = len(passage_results)
    passages_with_suited = [r for r in passage_results if r["n_suited_available"] > 0]
    n_passages_suited = len(passages_with_suited)

    total_suited_available = sum(r["n_suited_available"] for r in passages_with_suited)
    total_suited_captured = sum(r["n_suited_captured"] for r in passages_with_suited)
    covered_suited_passages = sum(r["suited_passage_covered"] for r in passages_with_suited)
    lost_suited_passages = sum(r["lost_suited_passage"] for r in passages_with_suited)
    lost_suited_opportunities = total_suited_available - total_suited_captured

    total_frames_captured = sum(r["n_frames_captured"] for r in passage_results)
    total_accepted = sum(r["n_accepted_captured"] for r in passage_results)
    total_rejected = sum(r["n_rejected_captured"] for r in passage_results)

    total_time_ms = sum(r["total_time_ms"] for r in passage_results)
    total_time_low_ms = sum(r["time_low_ms"] for r in passage_results)
    total_time_high_ms = sum(r["time_high_ms"] for r in passage_results)

    transitions_l2h = sum(r["transitions_low_to_high"] for r in passage_results)
    transitions_h2l = sum(r["transitions_high_to_low"] for r in passage_results)

    all_high_durations = [d for r in passage_results for d in r["high_episodes_durations_ms"]]
    mean_high_dur = float(np.mean(all_high_durations)) if all_high_durations else 0.0
    median_high_dur = float(np.median(all_high_durations)) if all_high_durations else 0.0
    p95_high_dur = float(np.percentile(all_high_durations, 95)) if all_high_durations else 0.0

    total_hold_prevented = sum(r["hold_prevented_downshift_count"] for r in passage_results)
    total_hold_suited_recovered = sum(r["hold_recovered_suited_frames"] for r in passage_results)

    return {
        "N": n_hold,
        "total_passages": total_passages,
        "passages_with_suited": n_passages_suited,
        "suited_passage_coverage_pct": (covered_suited_passages / n_passages_suited * 100.0),
        "covered_suited_passages": covered_suited_passages,
        "lost_suited_passages": lost_suited_passages,
        "total_suited_available": total_suited_available,
        "total_suited_captured": total_suited_captured,
        "suited_frame_retention_pct": (total_suited_captured / total_suited_available * 100.0),
        "lost_suited_opportunities": lost_suited_opportunities,
        "total_frames_captured": total_frames_captured,
        "total_accepted_captured": total_accepted,
        "total_rejected_captured": total_rejected,
        "total_time_low_s": total_time_low_ms / 1000.0,
        "pct_time_low": (total_time_low_ms / total_time_ms * 100.0),
        "total_time_high_s": total_time_high_ms / 1000.0,
        "pct_time_high": (total_time_high_ms / total_time_ms * 100.0),
        "transitions_low_to_high": transitions_l2h,
        "transitions_high_to_low": transitions_h2l,
        "mean_high_episode_duration_ms": mean_high_dur,
        "median_high_episode_duration_ms": median_high_dur,
        "p95_high_episode_duration_ms": p95_high_dur,
        "hold_prevented_downshift_count": total_hold_prevented,
        "hold_recovered_suited_frames": total_hold_suited_recovered,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    passage_ids = base.load_cohort(base.DEFAULT_COHORT_METRICS)
    indexes = base.load_indexes(base.DEFAULT_DATA_ROOT, passage_ids)
    feature_rows = read_rows(FEATURES_CSV)
    features = {(row["passage_id"], int(row["capture_index"])): row for row in feature_rows}
    raw_scores = existing_score_lookup()

    visual_states, threshold, direction = get_visual_post_states(indexes, features, raw_scores)
    selection_decisions = load_materialized_selection_decisions()

    # 1. Diagnóstico das 6 Passagens Perdidas no Cenário A
    lost_ids = ["0009vd", "0034az", "0060az", "0275", "0987", "0995"]
    detailed_traces = []

    for tag in lost_ids:
        frames = indexes[tag]
        for i, f in enumerate(frames):
            c_idx = i + 1
            t_ms = float(f["relative_time_ms"])
            lbl = f["label"]
            feat = features[(tag, c_idx)]
            p99 = float(feat["depth_p99_mm"])
            frac = float(feat["fraction_ge_2500mm"])
            inv = (p99 >= P99_YOUDEN) and (frac >= FRACTION_GE_2500_YOUDEN)
            v_post = visual_states[tag][i]
            sel = selection_decisions[tag][c_idx]

            detailed_traces.append({
                "passage_id": tag,
                "capture_index": c_idx,
                "timestamp_ms": t_ms,
                "human_label": lbl,
                "invalid": inv,
                "depth_p99_mm": p99,
                "fraction_ge_2500mm": frac,
                "visual_active_post": v_post,
                "selection_suitable": sel,
            })

    with (OUTPUT_DIR / "lost_passages_full_native_traces.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_traces[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_traces)

    # 2. Executar Replay Cenário B (Visual Decoupled Full-Rate) para N=0, N=2, N=3 com LOW=2 FPS
    results_scenario_b: dict[int, list[dict]] = {}
    summaries_scenario_b: dict[int, dict[str, Any]] = {}

    for n in (0, 2, 3):
        p_res = [
            simulate_decoupled_visual_adaptive(
                tag, indexes[tag], visual_states[tag], selection_decisions[tag], n_hold=n, low_fps=2.0
            )
            for tag in passage_ids
        ]
        results_scenario_b[n] = p_res
        summaries_scenario_b[n] = aggregate_cohort(p_res, n)

    with (OUTPUT_DIR / "scenario_b_decoupled_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries_scenario_b[0].keys()))
        writer.writeheader()
        for n in (0, 2, 3):
            writer.writerow(summaries_scenario_b[n])

    # 3. Executar Replay com LOW = 3 FPS no Cenário A e B para comparação
    p_res_low3_scen_b = [
        simulate_decoupled_visual_adaptive(
            tag, indexes[tag], visual_states[tag], selection_decisions[tag], n_hold=2, low_fps=3.0
        )
        for tag in passage_ids
    ]
    summary_low3_scen_b = aggregate_cohort(p_res_low3_scen_b, 2)

    print("=== CENÁRIO B (Visual Decoupled Full-Rate) - RESUMO ===")
    print(json.dumps(summaries_scenario_b, indent=2))

    print("\n=== CENÁRIO B (LOW = 3 FPS, N = 2) ===")
    print(json.dumps(summary_low3_scen_b, indent=2))


if __name__ == "__main__":
    main()
