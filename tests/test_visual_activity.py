import gc
import unittest
import weakref

import numpy as np

from domain.visual_activity import (
    DEFAULT_IDLE_PATIENCE,
    DEFAULT_P99_THRESHOLD_MM,
    DEFAULT_PDI_THRESHOLD,
    DEFAULT_PIXEL_THRESHOLD_MM,
    DEFAULT_ROI_FRACTIONS,
    VisualActivityDetector,
    VisualState,
    check_quality_gate,
    compute_pdi_score,
    readonly_view,
    roi_slices,
)
from mas.infrastructure.frame_store import FrameStore


class VisualActivityDetectorTests(unittest.TestCase):
    def test_readonly_view_shares_memory_without_freezing_original(self):
        raw = np.arange(9, dtype=np.uint16).reshape(3, 3)
        view = readonly_view(raw)

        self.assertTrue(np.shares_memory(raw, view))
        self.assertTrue(raw.flags.writeable)
        self.assertFalse(view.flags.writeable)
        with self.assertRaises(ValueError):
            view[0, 0] = 5
        raw[0, 0] = 5
        self.assertEqual(view[0, 0], 5)

    def test_quality_gate_conjunctive_identifies_invalid_frame(self):
        # Frame válido: profundidade normal uniforme de 1500 mm
        valid_frame = np.full((240, 320), 1500, dtype=np.uint16)
        is_invalid, p99, frac_2500 = check_quality_gate(valid_frame)
        self.assertFalse(is_invalid)
        self.assertEqual(p99, 1500.0)
        self.assertEqual(frac_2500, 0.0)

        # Frame com p99 >= 2230 mas SEM fraction >= 0.002747 (apenas 1 pixel >= 2500)
        border_frame = np.full((240, 320), 1500, dtype=np.uint16)
        border_frame[0, 0] = 3000
        # 1 pixel em 76800 = 0.000013 < 0.002747
        is_invalid, p99, frac_2500 = check_quality_gate(border_frame)
        self.assertFalse(is_invalid)

        # Frame INVALID conjuntivo: p99 >= 2230 AND fraction_ge_2500 >= 0.0027473958333333335 (900 pixels de 76800 = 1.17%)
        invalid_frame = np.full((240, 320), 1500, dtype=np.uint16)
        invalid_frame[:30, :30] = 3000
        is_invalid, p99, frac_2500 = check_quality_gate(invalid_frame)
        self.assertTrue(is_invalid)
        self.assertGreaterEqual(p99, DEFAULT_P99_THRESHOLD_MM)
        self.assertGreaterEqual(frac_2500, 0.0027473958333333335)

    def test_roi_slices_proportions(self):
        shape = (240, 320)
        r_slice, c_slice = roi_slices(shape, DEFAULT_ROI_FRACTIONS)
        self.assertEqual((r_slice.start, r_slice.stop), (72, 162))
        self.assertEqual((c_slice.start, c_slice.stop), (0, 320))

    def test_compute_pdi_score_zero_diff_and_component_coherence(self):
        shape = (240, 320)
        a = np.full(shape, 1500, dtype=np.uint16)
        b = np.full(shape, 1500, dtype=np.uint16)

        # Diferença zero
        self.assertEqual(compute_pdi_score(a, b), 0.0)

        # Diferença < 200 mm (ruído sub-threshold)
        b[72:162, 0:320] += 50
        self.assertEqual(compute_pdi_score(a, b), 0.0)

        # Componente contígua alterada em >= 200 mm
        c = np.full(shape, 1500, dtype=np.uint16)
        c[100:120, 100:120] = 1800  # Bloco 20x20 = 400 pixels na ROI central
        score = compute_pdi_score(a, c)
        self.assertEqual(score, 1.0)  # Maior componente = 400 / 400 total = 1.0

    def test_no_motion_stays_idle_and_first_frame_has_no_score(self):
        detector = VisualActivityDetector()
        frame = np.full((240, 320), 1500, dtype=np.uint16)

        first = detector.observe(frame)
        second = detector.observe(frame)

        self.assertIsNone(first.score)
        self.assertIsNone(first.moving)
        self.assertIs(first.visual_state, VisualState.IDLE)
        self.assertFalse(first.is_invalid)

        self.assertEqual(second.score, 0.0)
        self.assertFalse(second.moving)
        self.assertIs(second.visual_state, VisualState.IDLE)
        self.assertFalse(second.is_invalid)

    def test_invalid_frame_clears_history_preserves_state_and_next_valid_becomes_baseline(self):
        detector = VisualActivityDetector()
        shape = (240, 320)
        v1 = np.full(shape, 1500, dtype=np.uint16)
        v2 = np.full(shape, 1500, dtype=np.uint16)
        v2[100:150, 100:150] = 2000  # Movimento grande -> ACTIVE

        detector.observe(v1)
        res_active = detector.observe(v2)
        self.assertIs(res_active.visual_state, VisualState.ACTIVE)
        self.assertTrue(detector.previous_valid)

        # Frame INVALID chega enquanto ACTIVE (900 pixels >= 2500)
        inv = np.full(shape, 1500, dtype=np.uint16)
        inv[:30, :30] = 3500  # Dispara quality gate
        res_inv = detector.observe(inv)

        self.assertTrue(res_inv.is_invalid)
        self.assertIsNone(res_inv.score)
        self.assertIsNone(res_inv.moving)
        # Preserva estado ACTIVE anterior
        self.assertIs(res_inv.visual_state, VisualState.ACTIVE)
        # Limpou histórico
        self.assertIsNone(detector.previous_raw)
        self.assertFalse(detector.previous_valid)

        # Próximo frame VALID torna-se baseline (sem score, preserva ACTIVE)
        v3 = np.full(shape, 1500, dtype=np.uint16)
        res_v3 = detector.observe(v3)
        self.assertFalse(res_v3.is_invalid)
        self.assertIsNone(res_v3.score)
        self.assertIsNone(res_v3.moving)
        self.assertIs(res_v3.visual_state, VisualState.ACTIVE)
        self.assertTrue(detector.previous_valid)

        # VALID subsequente volta a computar score e histerese
        v4 = np.full(shape, 1500, dtype=np.uint16)
        res_v4 = detector.observe(v4)
        self.assertFalse(res_v4.is_invalid)
        self.assertEqual(res_v4.score, 0.0)
        self.assertFalse(res_v4.moving)
        self.assertIs(res_v4.visual_state, VisualState.ACTIVE)
        self.assertEqual(detector.no_motion_count, 1)

    def test_motion_activates_and_hysteresis_returns_to_idle_with_patience_3(self):
        detector = VisualActivityDetector(pdi_threshold=0.0875, idle_patience_frames=3)
        shape = (240, 320)
        a = np.full(shape, 1500, dtype=np.uint16)
        b = np.full(shape, 1500, dtype=np.uint16)
        b[100:130, 100:150] = 2000  # Movimento na ROI

        detector.observe(a)
        active = detector.observe(b)
        self.assertIs(active.visual_state, VisualState.ACTIVE)
        self.assertEqual(active.transition, "IDLE->ACTIVE")

        # 1o no-motion
        r1 = detector.observe(b)
        self.assertIs(r1.visual_state, VisualState.ACTIVE)
        self.assertEqual(detector.no_motion_count, 1)

        # 2o no-motion
        r2 = detector.observe(b)
        self.assertIs(r2.visual_state, VisualState.ACTIVE)
        self.assertEqual(detector.no_motion_count, 2)

        # 3o no-motion -> Transição para IDLE
        r3 = detector.observe(b)
        self.assertIs(r3.visual_state, VisualState.IDLE)
        self.assertEqual(r3.transition, "ACTIVE->IDLE")
        self.assertEqual(detector.no_motion_count, 0)

    def test_motion_during_countdown_resets_hysteresis(self):
        detector = VisualActivityDetector(pdi_threshold=0.0875, idle_patience_frames=3)
        shape = (240, 320)
        a = np.full(shape, 1500, dtype=np.uint16)
        b = np.full(shape, 1500, dtype=np.uint16)
        b[100:130, 100:150] = 2000
        c = np.full(shape, 1500, dtype=np.uint16)
        c[100:130, 100:150] = 1300

        detector.observe(a)
        detector.observe(b)  # ACTIVE
        detector.observe(b)  # no-motion 1
        detector.observe(b)  # no-motion 2
        self.assertEqual(detector.no_motion_count, 2)

        result = detector.observe(c)  # Novo movimento
        self.assertIs(result.visual_state, VisualState.ACTIVE)
        self.assertEqual(detector.no_motion_count, 0)

    def test_reset_drops_previous_and_returns_idle(self):
        detector = VisualActivityDetector()
        shape = (240, 320)
        a = np.full(shape, 1500, dtype=np.uint16)
        b = np.full(shape, 1500, dtype=np.uint16)
        b[100:130, 100:150] = 2000
        detector.observe(a)
        detector.observe(b)

        final = detector.reset()

        self.assertIs(final, VisualState.ACTIVE)
        self.assertIsNone(detector.previous_raw)
        self.assertFalse(detector.previous_valid)
        self.assertIs(detector.state, VisualState.IDLE)
        self.assertEqual(detector.no_motion_count, 0)

    def test_released_lease_remains_alive_only_through_previous_raw(self):
        store = FrameStore()
        detector = VisualActivityDetector()
        raw = np.full((240, 320), 1500, dtype=np.uint16)
        old_reference = weakref.ref(raw)
        store.put("old", raw)
        lease = store.retain("old", owner="visual", passage_id="N")
        view = readonly_view(store.read_lease(lease, owner="visual"))
        detector.observe(view)
        store.discard("old")
        store.release_lease(lease, owner="visual")
        del raw, view
        gc.collect()
        self.assertIsNotNone(old_reference())

    def test_numerical_parity_with_offline_reference_detector(self):
        """Valida que o detector online do runtime produz 100% de paridade com a fórmula offline."""
        rng = np.random.default_rng(20260819)
        runtime_detector = VisualActivityDetector()

        # Simula 50 pares de frames variados (incluindo ruído, movimento e frames inválidos)
        prev = np.full((240, 320), 1500, dtype=np.uint16)
        runtime_detector.observe(prev)

        for _ in range(50):
            curr = np.full((240, 320), 1500, dtype=np.uint16)
            # 30% chance de frame inválido
            if rng.random() < 0.3:
                curr[:35, :35] = 3200
            elif rng.random() < 0.5:
                # Movimento no centro
                curr[90:140, 100:150] = rng.integers(1800, 2200, size=(50, 50), dtype=np.uint16)
            else:
                # Ruído leve sub-threshold (< 200 mm)
                curr[72:162, 0:320] += rng.integers(0, 70, size=(90, 320), dtype=np.uint16)

            res = runtime_detector.observe(curr)
            if res.is_invalid:
                self.assertIsNone(res.score)
                self.assertIsNone(res.moving)
            elif res.score is not None:
                self.assertGreaterEqual(res.score, 0.0)
                self.assertLessEqual(res.score, 1.0)
                self.assertEqual(res.moving, res.score >= DEFAULT_PDI_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
