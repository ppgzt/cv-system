import gc
import unittest
import weakref

import numpy as np

from domain.visual_activity import (
    VisualActivityDetector,
    VisualState,
    mean_absolute_depth_difference,
    readonly_view,
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

    def test_uint16_difference_does_not_underflow(self):
        high = np.array([[1000]], dtype=np.uint16)
        low = np.array([[10]], dtype=np.uint16)
        self.assertEqual(mean_absolute_depth_difference(high, low), 990.0)
        self.assertEqual(mean_absolute_depth_difference(low, high), 990.0)

    def test_no_motion_stays_idle_and_first_frame_has_no_mad(self):
        detector = VisualActivityDetector(10.0, 3)
        frame = np.full((2, 2), 100, dtype=np.uint16)

        first = detector.observe(frame)
        second = detector.observe(frame)

        self.assertIsNone(first.mad)
        self.assertIsNone(first.moving)
        self.assertIs(first.visual_state, VisualState.IDLE)
        self.assertEqual(second.mad, 0.0)
        self.assertFalse(second.moving)
        self.assertIs(second.visual_state, VisualState.IDLE)

    def test_motion_activates_and_hysteresis_returns_to_idle(self):
        detector = VisualActivityDetector(10.0, 3)
        a = np.zeros((2, 2), dtype=np.uint16)
        b = np.full((2, 2), 20, dtype=np.uint16)

        detector.observe(a)
        active = detector.observe(b)
        self.assertIs(active.visual_state, VisualState.ACTIVE)
        self.assertEqual(active.transition, "IDLE->ACTIVE")

        self.assertIs(detector.observe(b).visual_state, VisualState.ACTIVE)
        self.assertIs(detector.observe(b).visual_state, VisualState.ACTIVE)
        idle = detector.observe(b)
        self.assertIs(idle.visual_state, VisualState.IDLE)
        self.assertEqual(idle.transition, "ACTIVE->IDLE")

    def test_motion_during_countdown_resets_hysteresis(self):
        detector = VisualActivityDetector(10.0, 3)
        a = np.zeros((2, 2), dtype=np.uint16)
        b = np.full((2, 2), 20, dtype=np.uint16)
        c = np.full((2, 2), 40, dtype=np.uint16)
        detector.observe(a)
        detector.observe(b)
        detector.observe(b)
        detector.observe(b)

        result = detector.observe(c)
        self.assertIs(result.visual_state, VisualState.ACTIVE)
        self.assertEqual(detector.no_motion_count, 0)

    def test_reset_drops_previous_and_returns_idle(self):
        detector = VisualActivityDetector(1.0, 2)
        detector.observe(np.zeros((1, 1), dtype=np.uint16))
        detector.observe(np.ones((1, 1), dtype=np.uint16) * 2)

        final = detector.reset()

        self.assertIs(final, VisualState.ACTIVE)
        self.assertIsNone(detector.previous_raw)
        self.assertIs(detector.state, VisualState.IDLE)
        self.assertEqual(detector.no_motion_count, 0)

    def test_released_lease_remains_alive_only_through_previous_raw(self):
        store = FrameStore()
        detector = VisualActivityDetector(1.0, 1)
        raw = np.array([[0, 1], [2, 3]], dtype=np.uint16)
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

        detector.observe(np.full((2, 2), 10, dtype=np.uint16))
        gc.collect()
        self.assertIsNone(old_reference())


if __name__ == "__main__":
    unittest.main()
