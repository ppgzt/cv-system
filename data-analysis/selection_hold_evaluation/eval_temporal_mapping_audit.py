#!/usr/bin/env python3
"""Auditoria Metodológica do Mapeamento Temporal: Nearest-Index vs Previous/Floor Causal.

Este script realiza:
1. Auditoria de deadlines e erros temporais de nearest_index() para LOW=4 e LOW=5;
2. Inspeção causal detalhada das passagens 0508, 0987, 0972, 0974 na região de entrada;
3. Avaliação de robustez da regra causal estrita (previous/floor: timestamp <= deadline);
4. Comparativo de coberturas, retenções e distribuições entre nearest e causal-floor.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
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
from domain.helpers.capture_schedule import nearest_index
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
)

OUTPUT_DIR = DATA_ANALYSIS / "selection_hold_evaluation" / "output"


def simulate_adaptive_with_mapping_policy(
    passage_id: str,
    frames: list[dict],
    visual_active_series: list[bool],
    selection_decisions: dict[int, bool],
    n_hold: int,
    low_fps: float,
    mapping_policy: str = "nearest",  # "nearest" ou "floor" (timestamp <= deadline)
) -> dict[str, Any]:
    n_frames = len(frames)
    timestamps = np.array([float(f["relative_time_ms"]) for f in frames])
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

    # Métricas de mapeamento temporal em LOW
    deadline_audit: list[dict[str, Any]] = []

    step_ms = 1000.0 / low_fps
    next_deadline_ms = timestamps[0]

    cursor = 0
    while cursor < n_frames:
        t_cursor = timestamps[cursor]

        admit = False
        if current_rate == "HIGH":
            admit = True
            selected_idx = cursor
        else:
            # Em LOW, verificamos se o cursor atual é o frame selecionado pelo deadline
            # Quando atingimos ou passamos o deadline
            if mapping_policy == "nearest":
                # Nearest index para next_deadline_ms
                # Para evitar loops infinitos ou retrocessos, o nearest deve ser >= cursor
                # mas na simulação sequencial, o cursor avança até o nearest_index
                target_idx = nearest_index(timestamps, next_deadline_ms)
                if cursor == target_idx:
                    admit = True
                    selected_idx = cursor
                    err = t_cursor - next_deadline_ms
                    deadline_audit.append({
                        "deadline_ms": next_deadline_ms,
                        "frame_idx": cursor + 1,
                        "frame_time_ms": t_cursor,
                        "signed_error_ms": err,
                        "abs_error_ms": abs(err),
                        "direction": "before" if err < -1e-4 else ("after" if err > 1e-4 else "exact"),
                        "label": labels[cursor],
                    })
                    next_deadline_ms = t_cursor + step_ms
                elif cursor < target_idx:
                    # Não atingiu o frame selecionado ainda
                    admit = False
                else:
                    # Cursor já passou do target_idx (se timestamps tiverem gaps)
                    if cursor > target_idx and (not captured_indices or captured_indices[-1] < target_idx):
                        admit = True
                        selected_idx = cursor
                        err = t_cursor - next_deadline_ms
                        deadline_audit.append({
                            "deadline_ms": next_deadline_ms,
                            "frame_idx": cursor + 1,
                            "frame_time_ms": t_cursor,
                            "signed_error_ms": err,
                            "abs_error_ms": abs(err),
                            "direction": "after",
                            "label": labels[cursor],
                        })
                        next_deadline_ms = t_cursor + step_ms
            elif mapping_policy == "floor":
                # Floor causal: último frame com timestamp <= deadline
                # Em stream online com relógio a 1000/fps:
                # O frame é emitido quando t_cursor >= next_deadline_ms (primeiro frame no/após deadline)
                # ou se usarmos o último frame do passado (floor):
                # No momento do deadline, o frame disponível mais recente é o último com t <= deadline
                # Para replay: quando t_cursor ultrapassa deadline, o frame imediatamente anterior (ou igual) é o emitido
                # Se next_deadline_ms coincide com o início da passagem:
                if cursor == 0:
                    admit = True
                    selected_idx = 0
                    err = 0.0
                    deadline_audit.append({
                        "deadline_ms": next_deadline_ms,
                        "frame_idx": 1,
                        "frame_time_ms": t_cursor,
                        "signed_error_ms": 0.0,
                        "abs_error_ms": 0.0,
                        "direction": "exact",
                        "label": labels[0],
                    })
                    next_deadline_ms = t_cursor + step_ms
                else:
                    # Encontrar o último frame com t <= next_deadline_ms
                    # Se t_cursor > next_deadline_ms, o frame cursor-1 foi o último no passado causal
                    # Verificamos se cursor é o primeiro frame após o deadline
                    if t_cursor >= next_deadline_ms - 1e-5:
                        # Se t_cursor é igual a next_deadline_ms, escolhe cursor
                        # Se t_cursor > next_deadline_ms, o floor causal é cursor-1 (se ainda não capturado) ou cursor
                        floor_idx = int(np.searchsorted(timestamps, next_deadline_ms, side="right")) - 1
                        floor_idx = max(0, floor_idx)
                        if cursor == floor_idx:
                            admit = True
                            selected_idx = cursor
                            err = timestamps[floor_idx] - next_deadline_ms
                            deadline_audit.append({
                                "deadline_ms": next_deadline_ms,
                                "frame_idx": cursor + 1,
                                "frame_time_ms": timestamps[floor_idx],
                                "signed_error_ms": err,
                                "abs_error_ms": abs(err),
                                "direction": "before" if err < -1e-4 else "exact",
                                "label": labels[cursor],
                            })
                            next_deadline_ms = timestamps[floor_idx] + step_ms
                        elif cursor > floor_idx and (not captured_indices or captured_indices[-1] < floor_idx):
                            admit = True
                            selected_idx = cursor
                            err = t_cursor - next_deadline_ms
                            deadline_audit.append({
                                "deadline_ms": next_deadline_ms,
                                "frame_idx": cursor + 1,
                                "frame_time_ms": t_cursor,
                                "signed_error_ms": err,
                                "abs_error_ms": abs(err),
                                "direction": "after",
                                "label": labels[cursor],
                            })
                            next_deadline_ms = t_cursor + step_ms

        if admit:
            captured_indices.append(cursor)
            capture_rates.append(current_rate)
            capture_times.append(t_cursor)

            v_active = visual_active_series[cursor]
            s_accepted = selection_decisions[cursor + 1]

            if n_hold > 0:
                if current_rate == "HIGH":
                    if s_accepted:
                        hold_active = True
                        consecutive_rejections = 0
                    else:
                        if hold_active:
                            consecutive_rejections += 1
                            if consecutive_rejections >= n_hold:
                                hold_active = False

            prev_rate = current_rate
            if v_active:
                target_rate = "HIGH"
            else:
                if current_rate == "HIGH" and hold_active and n_hold > 0:
                    target_rate = "HIGH"
                    hold_prevented_downshift_count += 1
                    if labels[cursor] == "suited":
                        hold_recovered_suited_frames += 1
                else:
                    target_rate = "LOW"
                    hold_active = False
                    consecutive_rejections = 0

            if prev_rate == "LOW" and target_rate == "HIGH":
                transitions_low_to_high += 1
                current_high_start_time = t_cursor
            elif prev_rate == "HIGH" and target_rate == "LOW":
                transitions_high_to_low += 1
                if current_high_start_time is not None:
                    high_episodes_duration_ms.append(
                        t_cursor - current_high_start_time
                    )
                    current_high_start_time = None
                next_deadline_ms = t_cursor + step_ms

            current_rate = target_rate

        cursor += 1

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
    suited_accepted_captured = sum(
        1 for idx in captured_suited if selection_decisions[idx + 1]
    )
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
        "n_suited_accepted_captured": suited_accepted_captured,
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
        "deadline_audit": deadline_audit,
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

    # 1. Mapeamento Temporal Audit para LOW=4 e LOW=5 (Nearest)
    audit_results = {}
    for low_fps in [4.0, 5.0]:
        all_deadlines = []
        for tag in passage_ids:
            res = simulate_adaptive_with_mapping_policy(
                tag, indexes[tag], visual_states[tag], selection_decisions[tag],
                n_hold=2, low_fps=low_fps, mapping_policy="nearest"
            )
            all_deadlines.extend(res["deadline_audit"])

        total_d = len(all_deadlines)
        before_d = sum(1 for d in all_deadlines if d["direction"] == "before")
        after_d = sum(1 for d in all_deadlines if d["direction"] == "after")
        exact_d = sum(1 for d in all_deadlines if d["direction"] == "exact")

        abs_errors = [d["abs_error_ms"] for d in all_deadlines]
        signed_errors = [d["signed_error_ms"] for d in all_deadlines]

        audit_results[low_fps] = {
            "total_deadlines": total_d,
            "before_count": before_d,
            "before_pct": before_d / total_d * 100.0,
            "after_count": after_d,
            "after_pct": after_d / total_d * 100.0,
            "exact_count": exact_d,
            "exact_pct": exact_d / total_d * 100.0,
            "abs_error_mean_ms": float(np.mean(abs_errors)),
            "abs_error_median_ms": float(np.median(abs_errors)),
            "abs_error_p95_ms": float(np.percentile(abs_errors, 95)),
            "abs_error_max_ms": float(np.max(abs_errors)),
            "signed_error_mean_ms": float(np.mean(signed_errors)),
            "signed_error_median_ms": float(np.median(signed_errors)),
            "signed_error_p95_ms": float(np.percentile(signed_errors, 95)),
            "signed_error_min_ms": float(np.min(signed_errors)),
            "signed_error_max_ms": float(np.max(signed_errors)),
        }

    with (OUTPUT_DIR / "nearest_deadline_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    # 2. Inspeção de 0508, 0987, 0972, 0974
    target_passages = ["0508", "0987", "0972", "0974"]
    inspection_records = {}

    for tag in target_passages:
        frames = indexes[tag]
        rec = {}
        for low_fps in [4.0, 5.0]:
            res = simulate_adaptive_with_mapping_policy(
                tag, frames, visual_states[tag], selection_decisions[tag],
                n_hold=2, low_fps=low_fps, mapping_policy="nearest"
            )
            rec[f"low_{int(low_fps)}fps"] = {
                "captured_indices_1based": [i + 1 for i in res["captured_indices"]],
                "suited_captured": res["n_suited_captured"],
                "total_accepted": res["n_accepted_captured"],
                "deadlines_in_entry": res["deadline_audit"][:8],
            }
        inspection_records[tag] = rec

    with (OUTPUT_DIR / "inspection_0508_0987_0972_0974.json").open("w", encoding="utf-8") as f:
        json.dump(inspection_records, f, indent=2)

    # 3. Análise de Robustez: Causal Floor (timestamp <= deadline)
    robustness_summaries = {}
    robustness_passage_rows = []

    for low_fps in [4.0, 5.0]:
        results_floor = [
            simulate_adaptive_with_mapping_policy(
                tag, indexes[tag], visual_states[tag], selection_decisions[tag],
                n_hold=2, low_fps=low_fps, mapping_policy="floor"
            )
            for tag in passage_ids
        ]
        cov = sum(1 for r in results_floor if r["suited_passage_covered"])
        suited_cap = sum(r["n_suited_captured"] for r in results_floor)
        tot_frames = sum(r["n_frames_captured"] for r in results_floor)
        tot_acc = sum(r["n_accepted_captured"] for r in results_floor)

        cov_thresholds_pred = {
            ">= 1": sum(1 for r in results_floor if r["n_accepted_captured"] >= 1),
            ">= 2": sum(1 for r in results_floor if r["n_accepted_captured"] >= 2),
            ">= 3": sum(1 for r in results_floor if r["n_accepted_captured"] >= 3),
            ">= 5": sum(1 for r in results_floor if r["n_accepted_captured"] >= 5),
        }

        cov_thresholds_suited = {
            ">= 1": sum(1 for r in results_floor if r["n_suited_captured"] >= 1),
            ">= 2": sum(1 for r in results_floor if r["n_suited_captured"] >= 2),
            ">= 3": sum(1 for r in results_floor if r["n_suited_captured"] >= 3),
            ">= 5": sum(1 for r in results_floor if r["n_suited_captured"] >= 5),
        }

        robustness_summaries[low_fps] = {
            "low_fps": low_fps,
            "coverage": cov,
            "coverage_pct": cov / 184.0 * 100.0,
            "suited_captured": suited_cap,
            "suited_retention_pct": suited_cap / 1655.0 * 100.0,
            "total_frames": tot_frames,
            "total_accepted": tot_acc,
            "reduction_vs_baseline_pct": (1.0 - (tot_frames / 13741.0)) * 100.0,
            "cov_thresholds_predictions": cov_thresholds_pred,
            "cov_thresholds_suited": cov_thresholds_suited,
            "lost_passages": [r["passage_id"] for r in results_floor if r["lost_suited_passage"]],
        }

    with (OUTPUT_DIR / "robustness_floor_summary.json").open("w", encoding="utf-8") as f:
        json.dump(robustness_summaries, f, indent=2)

    print("=== AUDITORIA METODOLÓGICA CONCLUÍDA ===")
    print("\n1. Auditoria de Deadlines (Nearest):")
    print(json.dumps(audit_results, indent=2))

    print("\n2. Auditoria de Robustez (Causal Floor: timestamp <= deadline):")
    print(json.dumps(robustness_summaries, indent=2))


if __name__ == "__main__":
    main()
