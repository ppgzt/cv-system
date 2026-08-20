#!/usr/bin/env python3
"""Investigação dos 25 Frames Suited Não Retidos em LOW=5 / N=2 e Auditoria de Causalidade do Detector."""

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

    low_fps = 5.0
    n_hold = 2

    # Executar simulação LOW=5 / N=2
    res_low5 = {
        tag: simulate_adaptive_passage(
            passage_id=tag,
            frames=indexes[tag],
            visual_active_series=visual_states[tag],
            selection_decisions=selection_decisions[tag],
            n_hold=n_hold,
            low_fps=low_fps,
        )
        for tag in passage_ids
    }

    lost_suited_records = []
    affected_passages = {}

    for tag in passage_ids:
        r = res_low5[tag]
        frames = indexes[tag]
        labels = [f["label"] for f in frames]
        timestamps = [float(f["relative_time_ms"]) for f in frames]
        v_states = visual_states[tag]
        cap_set = set(r["captured_indices"])

        # Encontrar suited perdidos nesta passagem
        suited_indices = [i for i, l in enumerate(labels) if l == "suited"]
        lost_in_passage = [i for i in suited_indices if i not in cap_set]

        if lost_in_passage:
            first_parcial = None
            first_suited = None
            last_suited = None
            for i, l in enumerate(labels):
                if l == "parcial" and first_parcial is None:
                    first_parcial = (i + 1, timestamps[i])
                if l == "suited":
                    if first_suited is None:
                        first_suited = (i + 1, timestamps[i])
                    last_suited = (i + 1, timestamps[i])

            # Primeiro frame efetivamente capturado que produz ACTIVE
            first_cap_active = None
            first_high_start_time = None
            for idx in r["captured_indices"]:
                if v_states[idx]:
                    first_cap_active = (idx + 1, timestamps[idx])
                    # Início efetivo de HIGH é no timestamp deste frame
                    first_high_start_time = timestamps[idx]
                    break

            affected_passages[tag] = {
                "passage_id": tag,
                "n_suited_available": len(suited_indices),
                "n_suited_lost": len(lost_in_passage),
                "first_parcial": first_parcial,
                "first_suited": first_suited,
                "first_cap_active": first_cap_active,
                "first_high_start_time": first_high_start_time,
                "last_suited": last_suited,
            }

            for idx in lost_in_passage:
                c_idx = idx + 1
                t_frame = timestamps[idx]
                v_post = v_states[idx]

                # Classificação causal
                # 1. antes do primeiro Visual ACTIVE no trace nativo
                # 2. trigger do ACTIVE (o frame exato onde o trace nativo vira ACTIVE)
                # 3. entre tick LOW e upshift (o Visual já estava ACTIVE no nativo, mas a captura estava em LOW aguardando tick)
                # 4. após downshift (o sistema subiu para HIGH e depois desceu para LOW antes desse suited)
                if first_cap_active is None or t_frame < first_cap_active[1]:
                    # Ocorrido antes do início de HIGH
                    if not v_post:
                        cat = "antes do primeiro Visual ACTIVE"
                    else:
                        # Se idx == primeiro ponto onde v_states vira True
                        if idx > 0 and not v_states[idx - 1] and v_states[idx]:
                            cat = "trigger do ACTIVE"
                        else:
                            cat = "entre tick LOW e upshift"
                else:
                    # Ocorrido após o início de HIGH
                    # Se ocorreu depois do primeiro HIGH, verificar se estava em LOW por downshift
                    cat = "após downshift"

                sel_val = selection_decisions[tag][c_idx]

                lost_suited_records.append({
                    "passage_id": tag,
                    "capture_index": c_idx,
                    "timestamp_ms": t_frame,
                    "human_label": labels[idx],
                    "selection_decision_native": "ACCEPTED" if sel_val else "REJECTED",
                    "visual_state_native": "ACTIVE" if v_post else "IDLE",
                    "classification": cat,
                    "first_suited_time_ms": first_suited[1] if first_suited else None,
                    "first_high_time_ms": first_high_start_time,
                })

    print(f"Total de suited não retidos em LOW=5: {len(lost_suited_records)}")
    print(f"Total de passagens afetadas: {len(affected_passages)}")

    with (OUTPUT_DIR / "lost_25_suited_low5_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(lost_suited_records[0].keys()))
        writer.writeheader()
        writer.writerows(lost_suited_records)

    with (OUTPUT_DIR / "affected_passages_low5_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "lost_suited_records": lost_suited_records,
                "affected_passages": affected_passages,
            },
            f,
            indent=2,
        )

    # 4. Separação de Suited Capturados vs Accepted vs Total Accepted em LOW=5
    total_suited_available = sum(r["n_suited_available"] for r in res_low5.values())
    total_suited_captured = sum(r["n_suited_captured"] for r in res_low5.values())
    
    # Suited capturados E aceitos
    total_suited_captured_accepted = 0
    total_accepted = 0
    total_rejected = 0

    for tag, r in res_low5.items():
        frames = indexes[tag]
        for idx in r["captured_indices"]:
            is_acc = selection_decisions[tag][idx + 1]
            if is_acc:
                total_accepted += 1
                if frames[idx]["label"] == "suited":
                    total_suited_captured_accepted += 1
            else:
                total_rejected += 1

    print("\n=== ESTATÍSTICAS DE CAPTURA E SELEÇÃO EM LOW=5 ===")
    print(f"Total Suited Disponíveis: {total_suited_available}")
    print(f"Total Suited Capturados: {total_suited_captured} (Perdidos na Captura: {total_suited_available - total_suited_captured})")
    print(f"Total Suited Capturados E Aceitos pelo Selection: {total_suited_captured_accepted} (Falsos Negativos do Selection em Suited Capturados: {total_suited_captured - total_suited_captured_accepted})")
    print(f"Total Accepted Capturados (Todas as classes): {total_accepted}")
    print(f"Total Rejected Capturados (Todas as classes): {total_rejected}")


if __name__ == "__main__":
    main()
