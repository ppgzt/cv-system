#!/usr/bin/env python3
"""Replay Adaptativo Causal com Deteção Visual Recalculada Online sobre Frames Capturados.

Este módulo corrige o problema metodológico de estado pré-computado:
1. Cada passagem instancia um detector Visual independente;
2. Somente frames efetivamente capturados são apresentados ao detector;
3. Quality gate conjuntivo reinicia o histórico temporal mantendo o estado visual;
4. Decisões do Selection e Selection Hold (N=2) atuam apenas sobre frames admitidos;
5. Avalia LOW=4 e LOW=5 sob causalidade online estrita.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
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
from eval_selection_hold import (
    load_materialized_selection_decisions,
)

OUTPUT_DIR = DATA_ANALYSIS / "selection_hold_evaluation" / "output"

# Parâmetros congelados do baseline PDI
ROI_B = (0.30, 0.70, 0.20, 0.80)
PIXEL_THRESHOLD_MM = 200.0
PDI_THRESHOLD = 0.08747855917667238
DIRECTION = 1.0
IDLE_PATIENCE = 3


@dataclass
class VisualObservationResult:
    score: float | None
    moving: bool | None
    visual_active: bool
    transition: str | None


class OnlineVisualDetector:
    """Detector Visual recalculado online, com histórico e histerese estritos."""

    def __init__(
        self,
        pdi_threshold: float = PDI_THRESHOLD,
        idle_patience: int = IDLE_PATIENCE,
        direction: float = DIRECTION,
    ):
        self.pdi_threshold = float(pdi_threshold)
        self.idle_patience = int(idle_patience)
        self.direction = float(direction)

        self.previous_raw: np.ndarray | None = None
        self.previous_valid: bool = False
        self.state: bool = False  # False = IDLE, True = ACTIVE
        self.no_motion_count: int = 0

    def observe(
        self, current_raw: np.ndarray, is_invalid: bool
    ) -> VisualObservationResult:
        previous_state = self.state

        if is_invalid:
            # INVALID: limpa histórico temporal, preserva estado atual
            self.previous_raw = None
            self.previous_valid = False
            self.no_motion_count = 0
            return VisualObservationResult(
                score=None,
                moving=None,
                visual_active=self.state,
                transition=None,
            )

        current = np.asarray(current_raw)

        if not self.previous_valid or self.previous_raw is None:
            # Primeiro VALID após início ou após INVALID: vira baseline
            self.previous_raw = current
            self.previous_valid = True
            return VisualObservationResult(
                score=None,
                moving=None,
                visual_active=self.state,
                transition=None,
            )

        # Segundo ou posterior frame VALID consecutivo: calcula PDI
        score = ablation.pdi_score(
            self.previous_raw,
            current,
            ablation.PHASE_ONE_VARIANTS["V0_baseline"],
        )
        self.previous_raw = current

        moving = (score * self.direction) >= self.pdi_threshold

        if moving:
            self.state = True
            self.no_motion_count = 0
        elif self.state:
            self.no_motion_count += 1
            if self.no_motion_count >= self.idle_patience:
                self.state = False
                self.no_motion_count = 0

        transition = None
        if self.state is not previous_state:
            transition = f"{'ACTIVE' if previous_state else 'IDLE'}->{'ACTIVE' if self.state else 'IDLE'}"

        return VisualObservationResult(
            score=score,
            moving=moving,
            visual_active=self.state,
            transition=transition,
        )


@dataclass
class SimulatedFrame:
    idx: int  # 1-indexed
    timestamp_ms: float
    label: str
    p99_mm: float
    frac_ge_2500: float
    depth_filename: str = ""
    depth_array: np.ndarray | None = None

    @property
    def is_invalid(self) -> bool:
        return (self.p99_mm >= P99_YOUDEN) and (
            self.frac_ge_2500 >= FRACTION_GE_2500_YOUDEN
        )


def simulate_online_adaptive_passage(
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

    captured_indices: list[int] = []
    capture_rates: list[str] = []
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

        admit = False
        if current_rate == "HIGH":
            admit = True
        else:
            if t_current >= next_low_scheduled_ms - 1e-5:
                admit = True
                next_low_scheduled_ms = t_current + (1000.0 / low_fps)

        if admit:
            captured_indices.append(frame_cursor)
            capture_rates.append(current_rate)
            capture_times.append(t_current)

            # 1. Carregar raw depth somente para o frame admitido
            if frame.depth_array is not None:
                raw_depth = frame.depth_array
            elif depth_loader is not None:
                raw_depth = depth_loader(passage_id, frame.depth_filename)
            else:
                raw_depth = base.read_depth(
                    base.DEFAULT_DATA_ROOT
                    / "DEPTH"
                    / passage_id
                    / frame.depth_filename
                )

            # 2. Visual Detector processa online o frame admitido
            obs = detector.observe(raw_depth, is_invalid=frame.is_invalid)
            v_active = obs.visual_active

            visual_observation_records.append({
                "capture_index": frame.idx,
                "timestamp_ms": t_current,
                "label": frame.label,
                "invalid": frame.is_invalid,
                "score": obs.score,
                "moving": obs.moving,
                "visual_active": v_active,
                "transition": obs.transition,
            })

            # 3. Selection avalia o frame admitido
            s_accepted = selection_decisions.get(frame.idx, False)

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

            # 4. Decisão causal de coordenação para frames futuros (> t_current)
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
        "visual_observations": visual_observation_records,
    }


def aggregate_online_cohort(passage_results: list[dict], low_fps: float) -> dict[str, Any]:
    total_passages = len(passage_results)
    passages_with_suited = [r for r in passage_results if r["n_suited_available"] > 0]
    n_passages_suited = len(passages_with_suited)

    total_suited_available = sum(r["n_suited_available"] for r in passages_with_suited)
    total_suited_captured = sum(r["n_suited_captured"] for r in passages_with_suited)
    total_suited_accepted = sum(r["n_suited_accepted_captured"] for r in passages_with_suited)
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

    pred_ge1 = sum(1 for r in passage_results if r["n_accepted_captured"] >= 1)
    pred_ge2 = sum(1 for r in passage_results if r["n_accepted_captured"] >= 2)
    pred_ge3 = sum(1 for r in passage_results if r["n_accepted_captured"] >= 3)
    pred_ge5 = sum(1 for r in passage_results if r["n_accepted_captured"] >= 5)

    suited_ge1 = sum(1 for r in passage_results if r["n_suited_captured"] >= 1)
    suited_ge2 = sum(1 for r in passage_results if r["n_suited_captured"] >= 2)
    suited_ge3 = sum(1 for r in passage_results if r["n_suited_captured"] >= 3)
    suited_ge5 = sum(1 for r in passage_results if r["n_suited_captured"] >= 5)

    return {
        "low_fps": low_fps,
        "total_passages": total_passages,
        "passages_with_suited": n_passages_suited,
        "suited_passage_coverage_pct": (covered_suited_passages / n_passages_suited * 100.0),
        "covered_suited_passages": covered_suited_passages,
        "lost_suited_passages": lost_suited_passages,
        "total_suited_available": total_suited_available,
        "total_suited_captured": total_suited_captured,
        "total_suited_accepted": total_suited_accepted,
        "suited_frame_retention_pct": (total_suited_captured / total_suited_available * 100.0),
        "lost_suited_opportunities": lost_suited_opportunities,
        "total_frames_captured": total_frames_captured,
        "total_accepted_captured": total_accepted,
        "total_rejected_captured": total_rejected,
        "reduction_vs_baseline_pct": (1.0 - (total_frames_captured / 13741.0)) * 100.0,
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
        "cov_thresholds_suited": {
            ">= 1": suited_ge1,
            ">= 2": suited_ge2,
            ">= 3": suited_ge3,
            ">= 5": suited_ge5,
        },
        "lost_passage_ids": [r["passage_id"] for r in passage_results if r["lost_suited_passage"]],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    passage_ids = base.load_cohort(base.DEFAULT_COHORT_METRICS)
    indexes = base.load_indexes(base.DEFAULT_DATA_ROOT, passage_ids)
    feature_rows = read_rows(FEATURES_CSV)
    features = {(row["passage_id"], int(row["capture_index"])): row for row in feature_rows}
    selection_decisions = load_materialized_selection_decisions()

    # Converter frames para SimulatedFrame
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

    # Cache de leitura de depth em memória por passagem
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
        print(f"Executando replay online estrito para LOW = {low_fps} FPS...")
        p_res = []
        for tag in passage_ids:
            res = simulate_online_adaptive_passage(
                passage_id=tag,
                frames=sim_frames_by_passage[tag],
                selection_decisions=selection_decisions[tag],
                n_hold=2,
                low_fps=low_fps,
                depth_loader=cached_depth_loader,
            )
            p_res.append(res)

        passage_results_by_fps[low_fps] = p_res
        agg = aggregate_online_cohort(p_res, low_fps)
        summaries[low_fps] = agg

        # Salvar CSV por passagem
        csv_file = OUTPUT_DIR / f"online_recalculated_n2_low_{int(low_fps)}fps_by_passage.csv"
        with csv_file.open("w", newline="", encoding="utf-8") as f:
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
                    "n_suited_accepted_captured",
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
            writer.writerows(p_res)

    with (OUTPUT_DIR / "online_recalculated_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("\n=== RESULTADOS DO REPLAY ONLINE RECALCULADO (N=2) ===")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
