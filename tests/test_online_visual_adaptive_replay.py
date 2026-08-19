#!/usr/bin/env python3
"""Testes unitários para validar a causalidade estrita e semântica do detector Visual Online no replay adaptativo."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ANALYSIS = REPO_ROOT / "data-analysis"
for path in (
    REPO_ROOT,
    DATA_ANALYSIS,
    DATA_ANALYSIS / "selection_hold_evaluation",
    DATA_ANALYSIS / "visual_event_quality_gate_audit",
    DATA_ANALYSIS / "visual_event_preprocessing_ablation",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_online_visual_replay import (
    OnlineVisualDetector,
    SimulatedFrame,
    simulate_online_adaptive_passage,
)


class TestOnlineVisualAdaptiveReplay(unittest.TestCase):
    def setUp(self):
        # Frame vazio 240x320
        self.shape = (240, 320)
        self.blank_frame = np.full(self.shape, 1500, dtype=np.uint16)
        # Frame com alteração grande na ROI B (0.30 a 0.70 vertical, 0.20 a 0.80 horizontal)
        self.moving_frame = self.blank_frame.copy()
        r_start, r_end = int(240 * 0.30), int(240 * 0.70)
        c_start, c_end = int(320 * 0.20), int(320 * 0.80)
        self.moving_frame[r_start:r_end, c_start:c_end] = 800  # Mudança de 700mm (> 200mm)

    def test_unadmitted_frame_does_not_affect_detector_state(self):
        """1. Inserir um frame não capturado entre dois capturados NÃO altera o estado Visual."""
        detector = OnlineVisualDetector(pdi_threshold=0.087478559, idle_patience=3)

        # Captura Frame 1 (Blank)
        res1 = detector.observe(self.blank_frame, is_invalid=False)
        self.assertFalse(res1.visual_active)
        self.assertIsNone(res1.score)

        # Frame não capturado ocorre no mundo físico (com movimento extremo), mas NÃO é apresentado ao detector
        unadmitted_moving = self.moving_frame.copy()

        # Captura Frame 2 (Blank)
        res2 = detector.observe(self.blank_frame, is_invalid=False)
        self.assertFalse(res2.visual_active)
        # Como ambos os capturados eram blank, o score entre eles é 0.0
        self.assertEqual(res2.score, 0.0)

    def test_detector_compares_consecutive_captured_frames_across_dataset_gaps(self):
        """2. O detector compara dois frames capturados consecutivos, mesmo com gap do dataset."""
        detector = OnlineVisualDetector(pdi_threshold=0.087478559, idle_patience=3)

        # Frame 1 capturado em t=0 (Blank)
        detector.observe(self.blank_frame, is_invalid=False)

        # Gap de 200ms no dataset (vários frames ocorreram no mundo físico)

        # Frame 2 capturado em t=200ms (Moving)
        res2 = detector.observe(self.moving_frame, is_invalid=False)
        self.assertIsNotNone(res2.score)
        self.assertGreater(res2.score, 0.087478559)
        self.assertTrue(res2.visual_active)

    def test_invalid_quality_gate_resets_temporal_history_and_preserves_state(self):
        """3. INVALID reseta histórico temporal (previous_raw=None) e preserva state."""
        detector = OnlineVisualDetector(pdi_threshold=0.087478559, idle_patience=3)

        # Estabelece ACTIVE
        detector.observe(self.blank_frame, is_invalid=False)
        res_active = detector.observe(self.moving_frame, is_invalid=False)
        self.assertTrue(res_active.visual_active)

        # Chega um frame INVALID (ex: ovelha cobrindo o sensor)
        res_invalid = detector.observe(self.blank_frame, is_invalid=True)
        # Deve preservar ACTIVE
        self.assertTrue(res_invalid.visual_active)
        self.assertIsNone(res_invalid.score)
        self.assertIsNone(detector.previous_raw)
        self.assertFalse(detector.previous_valid)
        self.assertEqual(detector.no_motion_count, 0)

    def test_new_valid_after_invalid_is_baseline_only(self):
        """4. Um novo VALID após INVALID é apenas baseline e não produz decisão temporal."""
        detector = OnlineVisualDetector(pdi_threshold=0.087478559, idle_patience=3)

        # Frame 1: INVALID
        detector.observe(self.blank_frame, is_invalid=True)

        # Frame 2: Primeiro VALID após INVALID
        res_baseline = detector.observe(self.moving_frame, is_invalid=False)
        self.assertIsNone(res_baseline.score)  # Não pode comparar com nada
        self.assertTrue(detector.previous_valid)
        self.assertIsNotNone(detector.previous_raw)

        # Frame 3: Segundo VALID (agora sim compara com Frame 2)
        res_next = detector.observe(self.moving_frame, is_invalid=False)
        self.assertIsNotNone(res_next.score)
        self.assertEqual(res_next.score, 0.0)  # Mesmo frame -> 0 movimento

    def test_unadmitted_frames_do_not_alter_selection_hold(self):
        """5. Frames não capturados não alteram Selection Hold."""
        frames = [
            SimulatedFrame(idx=1, timestamp_ms=0.0, label="background", p99_mm=1900.0, frac_ge_2500=0.0005, depth_array=self.blank_frame),
            SimulatedFrame(idx=2, timestamp_ms=100.0, label="parcial", p99_mm=1900.0, frac_ge_2500=0.0005, depth_array=self.moving_frame),
            SimulatedFrame(idx=3, timestamp_ms=200.0, label="suited", p99_mm=1900.0, frac_ge_2500=0.0005, depth_array=self.moving_frame),
            SimulatedFrame(idx=4, timestamp_ms=300.0, label="suited", p99_mm=1900.0, frac_ge_2500=0.0005, depth_array=self.moving_frame),
        ]
        # Frame 1 (DISCARD), Frame 2 (ACCEPTED no dataset nativo), Frame 3 (ACCEPTED), Frame 4 (ACCEPTED)
        selection_decisions = {1: False, 2: True, 3: True, 4: True}

        # Simulação com LOW=5 FPS (passo 200ms)
        # Frame 1 admitido em t=0 (DISCARD). Próximo agendado t=200ms (Frame 3).
        # Frame 2 (t=100ms) NÃO é capturado.
        res = simulate_online_adaptive_passage(
            passage_id="test",
            frames=frames,
            selection_decisions=selection_decisions,
            n_hold=2,
            low_fps=5.0,
        )
        # Frame 2 não foi capturado
        self.assertNotIn(1, res["captured_indices"])  # 0-indexed: index 1 é Frame 2
        self.assertIn(0, res["captured_indices"])      # Frame 1
        self.assertIn(2, res["captured_indices"])      # Frame 3


if __name__ == "__main__":
    unittest.main()
