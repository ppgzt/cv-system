#!/usr/bin/env python3
"""Avaliação e Replay Causal do Selection Hold (N=0, N=2, N=3).

Este script realiza a ablação offline isolando Visual Event + Selection Hold
sobre o cohort operacional completo de 184 passagens (13.741 frames).

Fontes de dados:
- Timestamps e Labels humanos: data/exp1/animal-tags/<tag>/simulation_index.json
- Decisões Visuais: Baseline PDI com Quality Gate Conjuntivo auditado
- Decisões Selection: MobileNetV2 precomputed da recon run nativa de 184 passagens
- Scheduler Adaptativo: LOW = 2 FPS (nearest-neighbor), HIGH = Full Rate / Original Timing
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
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
from domain.helpers.capture_schedule import nearest_index
from review_and_rerun_baseline import (
    build_baseline_series,
    existing_score_lookup,
)

OUTPUT_DIR = DATA_ANALYSIS / "selection_hold_evaluation" / "output"
RECON_METRICS = (
    REPO_ROOT
    / "recon_runs/recon_mas-single_pade_native_20260815_120525/mas-single_native_recon_r1/mas-single_pade_2026-08-15T12:38:04.754804/metrics.json"
)


def load_materialized_selection_decisions() -> dict[str, dict[int, bool]]:
    """Carrega as decisões reais do Selection (1.670 frames aceitos nas 184 passagens)."""
    with RECON_METRICS.open("r", encoding="utf-8") as f:
        data = json.load(f)
    animals = data["animals"]
    decisions: dict[str, dict[int, bool]] = {}
    for tag, info in animals.items():
        # capture_index no metrics.json é 1-indexed string
        suitable_indices = {int(k) for k in info.get("imgs", {}).keys()}
        total_frames = int(info["total_of_images"])
        decisions[tag] = {
            idx: (idx in suitable_indices) for idx in range(1, total_frames + 1)
        }
    return decisions


def get_visual_post_states(
    indexes: dict[str, list[dict]],
    features: dict[tuple[str, int], dict],
    raw_scores: dict[tuple[str, int], float],
) -> tuple[dict[str, list[bool]], float, float]:
    """Computa a série de estados visuais (IDLE=False, ACTIVE=True) no gate conjuntivo."""
    conjunctive_series = build_baseline_series(
        indexes, features, raw_scores, "conjunctive"
    )
    oracle_series = build_baseline_series(
        indexes, features, raw_scores, "oracle_label"
    )
    oracle_metrics = ablation.frame_metrics(indexes, oracle_series)
    threshold, direction = (
        oracle_metrics["threshold_directed"],
        oracle_metrics["direction"],
    )

    visual_states: dict[str, list[bool]] = {}
    for passage_id, frames in indexes.items():
        state, no_motion = False, 0
        post_states = []
        for index, score in enumerate(conjunctive_series[passage_id]):
            if math.isfinite(score):
                moving = score * direction >= threshold
                if moving:
                    state, no_motion = True, 0
                elif state:
                    no_motion += 1
                    if no_motion >= ablation.IDLE_PATIENCE:
                        state, no_motion = False, 0
            post_states.append(state)
        visual_states[passage_id] = post_states
    return visual_states, threshold, direction


def simulate_adaptive_passage(
    passage_id: str,
    frames: list[dict],
    visual_active_series: list[bool],
    selection_decisions: dict[int, bool],
    n_hold: int,
    low_fps: float = 2.0,
) -> dict[str, Any]:
    """Simula causalmente o scheduler adaptativo para uma passagem.

    Estados da taxa:
    - LOW (2 FPS): próximo frame agendado a t + 1000/low_fps ms
    - HIGH (Full Rate): admite todos os frames nativos
    """
    n_frames = len(frames)
    timestamps = [float(f["relative_time_ms"]) for f in frames]
    labels = [f["label"] for f in frames]

    # Estado inicial da captura: LOW (2 FPS)
    # A captura começa no primeiro frame nativo (t=timestamps[0])
    captured_indices: list[int] = []
    capture_rates: list[str] = []  # "LOW" ou "HIGH" por frame capturado
    capture_times: list[float] = []

    # Estado de controle
    current_rate = "LOW"
    hold_active = False
    consecutive_rejections = 0

    # Contadores de eventos
    hold_prevented_downshift_count = 0
    hold_recovered_suited_frames = 0
    transitions_low_to_high = 0
    transitions_high_to_low = 0

    # Rastreamento de episódios HIGH
    high_episodes_duration_ms: list[float] = []
    current_high_start_time: float | None = None

    # Tempo total em cada modo (calculado ao longo dos intervalos)
    time_low_ms = 0.0
    time_high_ms = 0.0

    # Loop causal de admissão de frames
    # current_frame_idx aponta para o próximo frame a ser avaliado
    frame_cursor = 0
    next_low_scheduled_ms = timestamps[0]

    while frame_cursor < n_frames:
        t_current = timestamps[frame_cursor]

        # 1. Determinar se frame_cursor deve ser admitido no modo atual
        admit = False
        if current_rate == "HIGH":
            admit = True
        else:
            # Modo LOW: admitir se atingiu ou passou o tempo agendado
            # Mapeamento nearest-neighbor a 2 FPS
            if t_current >= next_low_scheduled_ms - 1e-5:
                admit = True
                # Atualiza próximo agendamento LOW (500 ms)
                next_low_scheduled_ms = t_current + (1000.0 / low_fps)

        if admit:
            captured_indices.append(frame_cursor)
            capture_rates.append(current_rate)
            capture_times.append(t_current)

            # 2. Percepção: Visual Event e Selection observam o frame capturado
            # Visual Event:
            v_active = visual_active_series[frame_cursor]

            # Selection:
            # capture_index é 1-indexed
            s_accepted = selection_decisions[frame_cursor + 1]

            # Atualização do Selection Hold (se N > 0)
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
                else:
                    # Em LOW, Selection NÃO provoca upshift nem ativa hold
                    pass

            # 3. Política de Coordenação / Controle da Taxa
            # Decisão causal para o próximo instante
            prev_rate = current_rate

            if v_active:
                # Visual ACTIVE -> sempre sobe/mantém HIGH
                target_rate = "HIGH"
            else:
                # Visual IDLE
                if current_rate == "HIGH" and hold_active and n_hold > 0:
                    # Downshift vetado pelo Hold
                    target_rate = "HIGH"
                    hold_prevented_downshift_count += 1
                    if labels[frame_cursor] == "suited":
                        hold_recovered_suited_frames += 1
                else:
                    # Downshift efetivado para LOW
                    target_rate = "LOW"
                    hold_active = False
                    consecutive_rejections = 0

            # Gerenciar transições
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
                # Ao entrar em LOW, agenda o próximo tick LOW a 500 ms
                next_low_scheduled_ms = t_current + (1000.0 / low_fps)

            current_rate = target_rate

        # Avançar cursor
        frame_cursor += 1

    # Fechar último episódio HIGH se terminou em HIGH
    if current_rate == "HIGH" and current_high_start_time is not None:
        high_episodes_duration_ms.append(
            timestamps[-1] - current_high_start_time
        )

    # Cálculo dos tempos LOW e HIGH sobre a linha do tempo
    # Usando intervalos reais entre frames capturados e transições
    total_passage_time_ms = (
        timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    )

    # Integrar tempo por estado ao longo dos intervalos entre frames nativos
    # ou por intervalos de captura
    for i in range(len(timestamps) - 1):
        dt = timestamps[i + 1] - timestamps[i]
        # O estado ativo no intervalo [t_i, t_{i+1})
        # Determinado pelo modo em que o frame i estava operando
        # Procuramos o último frame capturado até o instante i
        # Para ser estrito:
        # Se frame i foi capturado em HIGH ou se estava em HIGH:
        pass

    # Uma forma mais padrão e robusta para tempo HIGH / LOW:
    # A soma das durações dos episódios HIGH
    time_high_ms = sum(high_episodes_duration_ms)
    time_low_ms = max(0.0, total_passage_time_ms - time_high_ms)

    # Métricas de suited e frames
    captured_set = set(captured_indices)
    suited_indices = [i for i, l in enumerate(labels) if l == "suited"]
    n_suited_available = len(suited_indices)
    captured_suited = [i for i in suited_indices if i in captured_set]
    n_suited_captured = len(captured_suited)

    # Decisões do selection nos frames capturados
    accepted_captured = sum(
        selection_decisions[idx + 1] for idx in captured_indices
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    passage_ids = base.load_cohort(base.DEFAULT_COHORT_METRICS)
    indexes = base.load_indexes(base.DEFAULT_DATA_ROOT, passage_ids)
    feature_rows = read_rows(FEATURES_CSV)
    features = {
        (row["passage_id"], int(row["capture_index"])): row
        for row in feature_rows
    }
    raw_scores = existing_score_lookup()

    print("Carregando decisões visuais (Gate Conjuntivo)...")
    visual_states, threshold, direction = get_visual_post_states(
        indexes, features, raw_scores
    )

    print("Carregando decisões materializadas do Selection...")
    selection_decisions = load_materialized_selection_decisions()

    # Avaliação para N in (0, 2, 3)
    results: dict[int, list[dict]] = {}
    summaries: dict[int, dict[str, Any]] = {}

    for n in (0, 2, 3):
        passage_results = []
        for tag in passage_ids:
            res = simulate_adaptive_passage(
                passage_id=tag,
                frames=indexes[tag],
                visual_active_series=visual_states[tag],
                selection_decisions=selection_decisions[tag],
                n_hold=n,
            )
            passage_results.append(res)
        results[n] = passage_results

        # Agregação do cohort
        total_passages = len(passage_results)
        passages_with_suited = [
            r for r in passage_results if r["n_suited_available"] > 0
        ]
        n_passages_suited = len(passages_with_suited)

        total_suited_available = sum(
            r["n_suited_available"] for r in passages_with_suited
        )
        total_suited_captured = sum(
            r["n_suited_captured"] for r in passages_with_suited
        )
        covered_suited_passages = sum(
            r["suited_passage_covered"] for r in passages_with_suited
        )
        lost_suited_passages = sum(
            r["lost_suited_passage"] for r in passages_with_suited
        )
        lost_suited_opportunities = (
            total_suited_available - total_suited_captured
        )

        total_frames_captured = sum(
            r["n_frames_captured"] for r in passage_results
        )
        total_accepted = sum(r["n_accepted_captured"] for r in passage_results)
        total_rejected = sum(r["n_rejected_captured"] for r in passage_results)

        total_time_ms = sum(r["total_time_ms"] for r in passage_results)
        total_time_low_ms = sum(r["time_low_ms"] for r in passage_results)
        total_time_high_ms = sum(r["time_high_ms"] for r in passage_results)

        transitions_l2h = sum(
            r["transitions_low_to_high"] for r in passage_results
        )
        transitions_h2l = sum(
            r["transitions_high_to_low"] for r in passage_results
        )

        all_high_durations = [
            d for r in passage_results for d in r["high_episodes_durations_ms"]
        ]
        mean_high_dur = (
            float(np.mean(all_high_durations)) if all_high_durations else 0.0
        )
        median_high_dur = (
            float(np.median(all_high_durations)) if all_high_durations else 0.0
        )
        p95_high_dur = (
            float(np.percentile(all_high_durations, 95))
            if all_high_durations
            else 0.0
        )

        total_hold_prevented = sum(
            r["hold_prevented_downshift_count"] for r in passage_results
        )
        total_hold_suited_recovered = sum(
            r["hold_recovered_suited_frames"] for r in passage_results
        )

        summaries[n] = {
            "N": n,
            "total_passages": total_passages,
            "passages_with_suited": n_passages_suited,
            "suited_passage_coverage_pct": (
                covered_suited_passages / n_passages_suited * 100.0
            ),
            "covered_suited_passages": covered_suited_passages,
            "lost_suited_passages": lost_suited_passages,
            "total_suited_available": total_suited_available,
            "total_suited_captured": total_suited_captured,
            "suited_frame_retention_pct": (
                total_suited_captured / total_suited_available * 100.0
            ),
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

    # Salvar resumos em CSV
    summary_path = OUTPUT_DIR / "selection_hold_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for n in (0, 2, 3):
            writer.writerow(summaries[n])

    # Salvar detalhes por passagem
    for n in (0, 2, 3):
        detail_path = OUTPUT_DIR / f"selection_hold_n{n}_by_passage.csv"
        with detail_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                k for k in results[n][0].keys() if k != "captured_indices"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results[n]:
                filtered = {k: v for k, v in row.items() if k in fieldnames}
                writer.writerow(filtered)

    # Diagnóstico comparativo por passagem
    diag_rows = []
    p0 = {r["passage_id"]: r for r in results[0]}
    p2 = {r["passage_id"]: r for r in results[2]}
    p3 = {r["passage_id"]: r for r in results[3]}

    for tag in passage_ids:
        r0, r2, r3 = p0[tag], p2[tag], p3[tag]
        suited_avail = r0["n_suited_available"]
        s0, s2, s3 = (
            r0["n_suited_captured"],
            r2["n_suited_captured"],
            r3["n_suited_captured"],
        )
        f0, f2, f3 = (
            r0["n_frames_captured"],
            r2["n_frames_captured"],
            r3["n_frames_captured"],
        )

        gain_suited_n2 = s2 - s0
        gain_suited_n3 = s3 - s0
        gain_suited_n3_over_n2 = s3 - s2

        gain_frames_n2 = f2 - f0
        gain_frames_n3 = f3 - f0

        # Classificação da passagem
        if s2 > s0 and s3 == s2:
            category = "GAINS_SUITED_N2_EQUALS_N3"
        elif s3 > s2 and s2 > s0:
            category = "GAINS_SUITED_N2_AND_MORE_N3"
        elif s3 > s0 and s2 == s0:
            category = "GAINS_SUITED_ONLY_N3"
        elif gain_frames_n2 > 0 and gain_suited_n2 == 0 and s3 == s2:
            category = "EXTRA_FRAMES_NO_SUITED_GAIN"
        elif f0 == f2 == f3 and s0 == s2 == s3:
            category = "IDENTICAL_ALL"
        else:
            category = "OTHER"

        diag_rows.append(
            {
                "passage_id": tag,
                "n_suited_available": suited_avail,
                "suited_n0": s0,
                "suited_n2": s2,
                "suited_n3": s3,
                "frames_n0": f0,
                "frames_n2": f2,
                "frames_n3": f3,
                "gain_suited_n2": gain_suited_n2,
                "gain_suited_n3": gain_suited_n3,
                "gain_suited_n3_over_n2": gain_suited_n3_over_n2,
                "gain_frames_n2": gain_frames_n2,
                "gain_frames_n3": gain_frames_n3,
                "category": category,
            }
        )

    diag_path = OUTPUT_DIR / "selection_hold_passage_diagnostics.csv"
    with diag_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diag_rows)

    print("Avaliação concluída com sucesso!")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
