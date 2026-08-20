#!/usr/bin/env python3
"""Análise de Redundância e Distribuição por Passagem: LOW=4 vs LOW=5 com Selection Hold N=2.

Este script investiga a localização e o impacto dos 16 frames suited adicionais de LOW=5
em relação a LOW=4, analisando:
1. Distribuição de passagens por número de oportunidades suited (0, 1, 2, 3, 4, 5+);
2. Distribuição de passagens por frames aceitos pelo Selection (0, 1, 2, 3, 4, 5+);
3. Coberturas com thresholds (>=1, >=2, >=3, >=5);
4. Lista ordenada de passagens com ganho em LOW=5;
5. Avaliação do ganho marginal (resgate de passagens críticas vs redundância adicional).
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

    n_hold = 2

    # Executar simulação para LOW = 4 e LOW = 5
    res_low4 = {
        tag: simulate_adaptive_passage(
            passage_id=tag,
            frames=indexes[tag],
            visual_active_series=visual_states[tag],
            selection_decisions=selection_decisions[tag],
            n_hold=n_hold,
            low_fps=4.0,
        )
        for tag in passage_ids
    }

    res_low5 = {
        tag: simulate_adaptive_passage(
            passage_id=tag,
            frames=indexes[tag],
            visual_active_series=visual_states[tag],
            selection_decisions=selection_decisions[tag],
            n_hold=n_hold,
            low_fps=5.0,
        )
        for tag in passage_ids
    }

    passage_rows = []
    for tag in passage_ids:
        r4 = res_low4[tag]
        r5 = res_low5[tag]
        frames = indexes[tag]
        labels = [f["label"] for f in frames]

        cap4_set = set(r4["captured_indices"])
        cap5_set = set(r5["captured_indices"])

        # Suited capturados
        suited_indices = [i for i, l in enumerate(labels) if l == "suited"]
        n_avail = len(suited_indices)

        suited_cap4 = sum(1 for i in suited_indices if i in cap4_set)
        suited_cap5 = sum(1 for i in suited_indices if i in cap5_set)

        # Suited capturados E aceitos pelo Selection
        suited_acc4 = sum(1 for i in suited_indices if i in cap4_set and selection_decisions[tag][i + 1])
        suited_acc5 = sum(1 for i in suited_indices if i in cap5_set and selection_decisions[tag][i + 1])

        # Total de frames aceitos pelo Selection (cada frame aceito gera 1 prediction de peso)
        tot_acc4 = sum(1 for i in cap4_set if selection_decisions[tag][i + 1])
        tot_acc5 = sum(1 for i in cap5_set if selection_decisions[tag][i + 1])

        gain_suited = suited_cap5 - suited_cap4
        gain_suited_acc = suited_acc5 - suited_acc4
        gain_tot_acc = tot_acc5 - tot_acc4

        passage_rows.append({
            "passage_id": tag,
            "n_suited_available": n_avail,
            "suited_captured_low4": suited_cap4,
            "suited_captured_low5": suited_cap5,
            "gain_suited": gain_suited,
            "suited_accepted_low4": suited_acc4,
            "suited_accepted_low5": suited_acc5,
            "gain_suited_accepted": gain_suited_acc,
            "total_accepted_low4": tot_acc4,
            "total_accepted_low5": tot_acc5,
            "gain_total_accepted": gain_tot_acc,
            "total_frames_low4": len(cap4_set),
            "total_frames_low5": len(cap5_set),
            "gain_total_frames": len(cap5_set) - len(cap4_set),
        })

    # Salvar CSV comparativo por passagem
    csv_file = OUTPUT_DIR / "redundancy_low4_vs_low5_by_passage.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(passage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(passage_rows)

    # 1. Distribuição de Suited Capturados
    def get_bin(val: int) -> str:
        if val == 0:
            return "0"
        elif val == 1:
            return "1"
        elif val == 2:
            return "2"
        elif val == 3:
            return "3"
        elif val == 4:
            return "4"
        else:
            return "5+"

    dist_suited_low4 = Counter(get_bin(r["suited_captured_low4"]) for r in passage_rows)
    dist_suited_low5 = Counter(get_bin(r["suited_captured_low5"]) for r in passage_rows)

    dist_acc_low4 = Counter(get_bin(r["total_accepted_low4"]) for r in passage_rows)
    dist_acc_low5 = Counter(get_bin(r["total_accepted_low5"]) for r in passage_rows)

    dist_suited_acc_low4 = Counter(get_bin(r["suited_accepted_low4"]) for r in passage_rows)
    dist_suited_acc_low5 = Counter(get_bin(r["suited_accepted_low5"]) for r in passage_rows)

    # 2. Coberturas por Thresholds (>=1, >=2, >=3, >=5)
    cov_thresholds_suited_low4 = {
        ">= 1": sum(1 for r in passage_rows if r["suited_captured_low4"] >= 1),
        ">= 2": sum(1 for r in passage_rows if r["suited_captured_low4"] >= 2),
        ">= 3": sum(1 for r in passage_rows if r["suited_captured_low4"] >= 3),
        ">= 5": sum(1 for r in passage_rows if r["suited_captured_low4"] >= 5),
    }

    cov_thresholds_suited_low5 = {
        ">= 1": sum(1 for r in passage_rows if r["suited_captured_low5"] >= 1),
        ">= 2": sum(1 for r in passage_rows if r["suited_captured_low5"] >= 2),
        ">= 3": sum(1 for r in passage_rows if r["suited_captured_low5"] >= 3),
        ">= 5": sum(1 for r in passage_rows if r["suited_captured_low5"] >= 5),
    }

    cov_thresholds_acc_low4 = {
        ">= 1": sum(1 for r in passage_rows if r["total_accepted_low4"] >= 1),
        ">= 2": sum(1 for r in passage_rows if r["total_accepted_low4"] >= 2),
        ">= 3": sum(1 for r in passage_rows if r["total_accepted_low4"] >= 3),
        ">= 5": sum(1 for r in passage_rows if r["total_accepted_low4"] >= 5),
    }

    cov_thresholds_acc_low5 = {
        ">= 1": sum(1 for r in passage_rows if r["total_accepted_low5"] >= 1),
        ">= 2": sum(1 for r in passage_rows if r["total_accepted_low5"] >= 2),
        ">= 3": sum(1 for r in passage_rows if r["total_accepted_low5"] >= 3),
        ">= 5": sum(1 for r in passage_rows if r["total_accepted_low5"] >= 5),
    }

    # 3. Lista de Passagens com Ganho em LOW=5
    gaining_passages = [r for r in passage_rows if r["gain_suited"] > 0]
    # Ordenar por: (1) menor suited_captured_low4, (2) passage_id
    gaining_passages.sort(key=lambda x: (x["suited_captured_low4"], x["passage_id"]))

    summary_out = {
        "total_passages": len(passage_rows),
        "total_suited_available": sum(r["n_suited_available"] for r in passage_rows),
        "total_suited_captured_low4": sum(r["suited_captured_low4"] for r in passage_rows),
        "total_suited_captured_low5": sum(r["suited_captured_low5"] for r in passage_rows),
        "total_gain_suited": sum(r["gain_suited"] for r in passage_rows),
        "total_suited_accepted_low4": sum(r["suited_accepted_low4"] for r in passage_rows),
        "total_suited_accepted_low5": sum(r["suited_accepted_low5"] for r in passage_rows),
        "total_gain_suited_accepted": sum(r["gain_suited_accepted"] for r in passage_rows),
        "total_accepted_low4": sum(r["total_accepted_low4"] for r in passage_rows),
        "total_accepted_low5": sum(r["total_accepted_low5"] for r in passage_rows),
        "total_gain_accepted": sum(r["gain_total_accepted"] for r in passage_rows),
        "dist_suited_captured_low4": dist_suited_low4,
        "dist_suited_captured_low5": dist_suited_low5,
        "dist_total_accepted_low4": dist_acc_low4,
        "dist_total_accepted_low5": dist_acc_low5,
        "dist_suited_accepted_low4": dist_suited_acc_low4,
        "dist_suited_accepted_low5": dist_suited_acc_low5,
        "cov_thresholds_suited_low4": cov_thresholds_suited_low4,
        "cov_thresholds_suited_low5": cov_thresholds_suited_low5,
        "cov_thresholds_accepted_low4": cov_thresholds_acc_low4,
        "cov_thresholds_accepted_low5": cov_thresholds_acc_low5,
        "gaining_passages": gaining_passages,
    }

    with (OUTPUT_DIR / "redundancy_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print("=== ANÁLISE DE REDUNDÂNCIA E DISTRIBUIÇÃO CONCLUÍDA ===")
    print(f"Total de passagens com ganho de suited em LOW=5: {len(gaining_passages)}")
    print(f"Soma dos ganhos em suited: {sum(r['gain_suited'] for r in gaining_passages)}")
    print("\nPassagens com ganho (ordenadas por menor suited em LOW=4):")
    for r in gaining_passages:
        print(f"  Passagem {r['passage_id']:8s} | Disponíveis={r['n_suited_available']:2d} | Suited Low4={r['suited_captured_low4']:2d} -> Low5={r['suited_captured_low5']:2d} (+{r['gain_suited']}) | Suited Acc Low4={r['suited_accepted_low4']:2d} -> Low5={r['suited_accepted_low5']:2d} (+{r['gain_suited_accepted']}) | Tot Acc Low4={r['total_accepted_low4']:2d} -> Low5={r['total_accepted_low5']:2d} (+{r['gain_total_accepted']})")


if __name__ == "__main__":
    main()
