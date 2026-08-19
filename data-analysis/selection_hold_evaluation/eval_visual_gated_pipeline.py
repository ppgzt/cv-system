#!/usr/bin/env python3
"""Avaliação do Pipeline Visual-Gated Causal com Admissão do Trigger Frame no Selection.

Este script implementa a semântica final de 4 estágios:
1. Aquisição Física (Sensor / Capture em LOW ou HIGH);
2. Decisão Visual (Detector Online PDI com Gate Conjuntivo);
3. Admissão no Ramo Pesado (Selection):
   - Todos os frames adquiridos em HIGH;
   - O frame trigger que provoca transição IDLE -> ACTIVE em LOW;
   - Frames adquiridos em LOW com Visual IDLE NÃO entram no Selection.
4. Aceitação pelo Selection (MobileNetV2) e Predição de Peso.
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
from eval_selection_hold import (
    load_materialized_selection_decisions,
)
from eval_online_visual_replay import (
    OnlineVisualDetector,
    SimulatedFrame,
)

OUTPUT_DIR = DATA_ANALYSIS / "selection_hold_evaluation" / "output"


def simulate_visual_gated_passage(
    passage_id: str,
    frames: list[SimulatedFrame],
    selection_decisions: dict[int, bool],
    n_hold: int,
    low_fps: float,
    depth_loader: Any = None,
) -> dict[str, Any]:
    n_frames = len(frames)
    timestamps = [f.timestamp_ms for f in frames]
    labels = [f.label for f in frames]

    detector = OnlineVisualDetector()

    physically_acquired_indices: list[int] = []
    admitted_to_selection_indices: list[int] = []
    capture_rates_at_acquisition: list[str] = []
    capture_times: list[float] = []

    visual_observation_records: list[dict[str, Any]] = []

    current_rate = "LOW"
    hold_active = False
    consecutive_rejections = 0

    hold_prevented_downshift_count = 0
    hold_recovered_suited_frames = 0
    transitions_low_to_high = 0
    transitions_high_to_low = 0

    high_episodes_duration_ms: list[float] = []
    current_high_start_time: float | None = None

    frame_cursor = 0
    next_low_scheduled_ms = timestamps[0]

    while frame_cursor < n_frames:
        t_current = timestamps[frame_cursor]
        frame = frames[frame_cursor]

        # 1. ESTÁGIO 1: AQUISIÇÃO FÍSICA (Capture Agent)
        admit_physical = False
        rate_at_physical = current_rate

        if current_rate == "HIGH":
            admit_physical = True
        else:
            if t_current >= next_low_scheduled_ms - 1e-5:
                admit_physical = True
                next_low_scheduled_ms = t_current + (1000.0 / low_fps)

        if admit_physical:
            physically_acquired_indices.append(frame_cursor)
            capture_rates_at_acquisition.append(rate_at_physical)
            capture_times.append(t_current)

            # Carregar depth
            if frame.depth_array is not None:
                raw_depth = frame.depth_array
            elif depth_loader is not None:
                raw_depth = depth_loader(passage_id, frame.depth_filename)
            else:
                raw_depth = base.read_depth(
                    base.DEFAULT_DATA_ROOT / "DEPTH" / passage_id / frame.depth_filename
                )

            # 2. ESTÁGIO 2: DECISÃO VISUAL (Visual Event Agent)
            obs = detector.observe(raw_depth, is_invalid=frame.is_invalid)
            v_active = obs.visual_active
            is_trigger = (obs.transition == "IDLE->ACTIVE")

            visual_observation_records.append({
                "capture_index": frame.idx,
                "timestamp_ms": t_current,
                "label": frame.label,
                "invalid": frame.is_invalid,
                "score": obs.score,
                "moving": obs.moving,
                "visual_active": v_active,
                "transition": obs.transition,
                "is_trigger": is_trigger,
            })

            # 3. ESTÁGIO 3: ADMISSÃO NO RAMO PESADO (Selection Agent)
            # Regra:
            # - Se adquirido em HIGH -> admitido no Selection
            # - Se adquirido em LOW e provocou IDLE->ACTIVE (Trigger) -> admitido no Selection!
            # - Se adquirido em LOW e permaneceu IDLE -> filtrado pelo Visual Gate (não vai ao Selection)
            admit_selection = False
            if rate_at_physical == "HIGH":
                admit_selection = True
            elif rate_at_physical == "LOW" and is_trigger:
                admit_selection = True

            if admit_selection:
                admitted_to_selection_indices.append(frame_cursor)

                # 4. ESTÁGIO 4: ACEITAÇÃO PELO SELECTION (MobileNetV2)
                s_accepted = selection_decisions.get(frame.idx, False)

                # Atualização do Selection Hold
                if n_hold > 0:
                    if s_accepted:
                        hold_active = True
                        consecutive_rejections = 0
                    else:
                        if hold_active:
                            consecutive_rejections += 1
                            if consecutive_rejections >= n_hold:
                                hold_active = False

            # 5. COORDENAÇÃO DE TAXA PARA FRAMES FUTUROS (> t_current)
            prev_rate = current_rate
            if v_active:
                target_rate = "HIGH"
            else:
                if current_rate == "HIGH" and hold_active and n_hold > 0:
                    target_rate = "HIGH"
                    hold_prevented_downshift_count += 1
                    if frame.label == "suited":
                        hold_recovered_suited_frames += 1
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

        frame_cursor += 1

    if current_rate == "HIGH" and current_high_start_time is not None:
        high_episodes_duration_ms.append(
            timestamps[-1] - current_high_start_time
        )

    total_passage_time_ms = (
        timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    )
    time_high_ms = sum(high_episodes_duration_ms)
    time_low_ms = max(0.0, total_passage_time_ms - time_high_ms)

    suited_indices = [i for i, l in enumerate(labels) if l == "suited"]
    n_suited_available = len(suited_indices)

    # Suited nos 4 estágios:
    phys_set = set(physically_acquired_indices)
    sel_set = set(admitted_to_selection_indices)

    suited_acquired = [i for i in suited_indices if i in phys_set]
    suited_forwarded_to_sel = [i for i in suited_indices if i in sel_set]
    suited_accepted_by_sel = [i for i in suited_forwarded_to_sel if selection_decisions[i + 1]]

    # Trigger frames que são suited
    suited_triggers = [
        obs["capture_index"] - 1
        for obs in visual_observation_records
        if obs["is_trigger"] and obs["label"] == "suited"
    ]

    # Total de predições no pipeline (todos os frames aceitos pelo Selection)
    total_accepted_predictions = sum(selection_decisions[idx + 1] for idx in admitted_to_selection_indices)

    return {
        "passage_id": passage_id,
        "n_frames_total": n_frames,
        "n_frames_physically_acquired": len(physically_acquired_indices),
        "n_frames_admitted_to_selection": len(admitted_to_selection_indices),
        "n_suited_available": n_suited_available,
        "n_suited_acquired": len(suited_acquired),
        "n_suited_triggers": len(suited_triggers),
        "n_suited_forwarded_to_selection": len(suited_forwarded_to_sel),
        "n_suited_accepted_by_selection": len(suited_accepted_by_sel),
        "n_suited_really_lost": n_suited_available - len(suited_acquired),
        "n_total_accepted_predictions": total_accepted_predictions,
        "suited_passage_covered": (len(suited_accepted_by_sel) > 0)
        if n_suited_available > 0
        else True,
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
        "physically_acquired_indices": physically_acquired_indices,
        "admitted_to_selection_indices": admitted_to_selection_indices,
        "visual_observations": visual_observation_records,
    }


def aggregate_visual_gated_cohort(passage_results: list[dict], low_fps: float) -> dict[str, Any]:
    total_passages = len(passage_results)
    passages_with_suited = [r for r in passage_results if r["n_suited_available"] > 0]
    n_passages_suited = len(passages_with_suited)

    total_suited_available = sum(r["n_suited_available"] for r in passages_with_suited)
    total_suited_acquired = sum(r["n_suited_acquired"] for r in passages_with_suited)
    total_suited_triggers = sum(r["n_suited_triggers"] for r in passages_with_suited)
    total_suited_forwarded = sum(r["n_suited_forwarded_to_selection"] for r in passages_with_suited)
    total_suited_accepted = sum(r["n_suited_accepted_by_selection"] for r in passages_with_suited)
    total_suited_really_lost = sum(r["n_suited_really_lost"] for r in passages_with_suited)

    covered_passages = sum(r["suited_passage_covered"] for r in passages_with_suited)

    total_physical_frames = sum(r["n_frames_physically_acquired"] for r in passage_results)
    total_selection_frames = sum(r["n_frames_admitted_to_selection"] for r in passage_results)
    total_accepted_preds = sum(r["n_total_accepted_predictions"] for r in passage_results)

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

    pred_ge1 = sum(1 for r in passage_results if r["n_total_accepted_predictions"] >= 1)
    pred_ge2 = sum(1 for r in passage_results if r["n_total_accepted_predictions"] >= 2)
    pred_ge3 = sum(1 for r in passage_results if r["n_total_accepted_predictions"] >= 3)
    pred_ge5 = sum(1 for r in passage_results if r["n_total_accepted_predictions"] >= 5)

    return {
        "low_fps": low_fps,
        "total_passages": total_passages,
        "passages_with_suited": n_passages_suited,
        "covered_suited_passages": covered_passages,
        "suited_passage_coverage_pct": (covered_passages / n_passages_suited * 100.0),
        "total_suited_available": total_suited_available,
        "total_suited_acquired": total_suited_acquired,
        "suited_acquisition_retention_pct": (total_suited_acquired / total_suited_available * 100.0),
        "total_suited_triggers": total_suited_triggers,
        "total_suited_forwarded_to_selection": total_suited_forwarded,
        "suited_forwarded_retention_pct": (total_suited_forwarded / total_suited_available * 100.0),
        "total_suited_accepted_by_selection": total_suited_accepted,
        "suited_accepted_retention_pct": (total_suited_accepted / total_suited_available * 100.0),
        "total_suited_really_lost": total_suited_really_lost,
        "total_frames_physically_acquired": total_physical_frames,
        "total_frames_admitted_to_selection": total_selection_frames,
        "selection_frames_reduction_vs_baseline_pct": (1.0 - (total_selection_frames / 13741.0)) * 100.0,
        "total_accepted_predictions": total_accepted_preds,
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
        "cov_thresholds_predictions": {
            ">= 1": pred_ge1,
            ">= 2": pred_ge2,
            ">= 3": pred_ge3,
            ">= 5": pred_ge5,
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    passage_ids = base.load_cohort(base.DEFAULT_COHORT_METRICS)
    indexes = base.load_indexes(base.DEFAULT_DATA_ROOT, passage_ids)
    feature_rows = read_rows(FEATURES_CSV)
    features = {(row["passage_id"], int(row["capture_index"])): row for row in feature_rows}
    selection_decisions = load_materialized_selection_decisions()

    sim_frames_by_passage: dict[str, list[SimulatedFrame]] = {}
    for tag in passage_ids:
        p_frames = []
        for idx, f in enumerate(indexes[tag], start=1):
            feat = features[(tag, idx)]
            p_frames.append(
                SimulatedFrame(
                    idx=idx,
                    timestamp_ms=float(f["relative_time_ms"]),
                    label=f["label"],
                    p99_mm=float(feat["depth_p99_mm"]),
                    frac_ge_2500=float(feat["fraction_ge_2500mm"]),
                    depth_filename=f["depth_filename"],
                )
            )
        sim_frames_by_passage[tag] = p_frames

    depth_cache: dict[tuple[str, str], np.ndarray] = {}

    def cached_depth_loader(pid: str, fname: str) -> np.ndarray:
        key = (pid, fname)
        if key not in depth_cache:
            depth_cache[key] = base.read_depth(
                base.DEFAULT_DATA_ROOT / "DEPTH" / pid / fname
            )
        return depth_cache[key]

    summaries = {}
    passage_results_by_fps = {}

    for low_fps in [4.0, 5.0]:
        print(f"Executando pipeline Visual-Gated para LOW = {low_fps} FPS...")
        p_res = []
        for tag in passage_ids:
            res = simulate_visual_gated_passage(
                passage_id=tag,
                frames=sim_frames_by_passage[tag],
                selection_decisions=selection_decisions[tag],
                n_hold=2,
                low_fps=low_fps,
                depth_loader=cached_depth_loader,
            )
            p_res.append(res)

        passage_results_by_fps[low_fps] = p_res
        agg = aggregate_visual_gated_cohort(p_res, low_fps)
        summaries[low_fps] = agg

        csv_file = OUTPUT_DIR / f"visual_gated_n2_low_{int(low_fps)}fps_by_passage.csv"
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "passage_id",
                    "n_frames_total",
                    "n_frames_physically_acquired",
                    "n_frames_admitted_to_selection",
                    "n_suited_available",
                    "n_suited_acquired",
                    "n_suited_triggers",
                    "n_suited_forwarded_to_selection",
                    "n_suited_accepted_by_selection",
                    "n_suited_really_lost",
                    "n_total_accepted_predictions",
                    "suited_passage_covered",
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
            writer.writerows(p_res)

    with (OUTPUT_DIR / "visual_gated_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("\n=== RESULTADOS DO PIPELINE VISUAL-GATED ===")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
