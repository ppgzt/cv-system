import unittest

import numpy as np

from domain.helpers.capture_schedule import (
    build_fixed_fps_schedule,
    nearest_index,
)


class CaptureScheduleTests(unittest.TestCase):
    def test_nearest_index_uses_previous_frame_on_tie(self):
        times = np.array([0.0, 1000.0, 2000.0])

        self.assertEqual(nearest_index(times, -1.0), 0)
        self.assertEqual(nearest_index(times, 500.0), 0)
        self.assertEqual(nearest_index(times, 501.0), 1)
        self.assertEqual(nearest_index(times, 3000.0), 2)

    def test_schedule_preserves_virtual_events_and_source_repetition(self):
        times = np.array([0.0, 400.0, 1000.0])

        schedule = build_fixed_fps_schedule(times, fps=4.0)

        self.assertEqual(
            [event.scheduled_capture_time_ms for event in schedule],
            [0.0, 250.0, 500.0, 750.0, 1000.0],
        )
        self.assertEqual([event.source_index for event in schedule], [0, 1, 1, 2, 2])

    def test_non_multiple_duration_stops_at_real_limit_not_next_tick(self):
        times = np.array([0.0, 5000.0, 5300.0])

        schedule = build_fixed_fps_schedule(times, fps=2.0)

        self.assertEqual(schedule[-1].scheduled_capture_time_ms, 5000.0)
        self.assertNotIn(
            5500.0, [event.scheduled_capture_time_ms for event in schedule]
        )


if __name__ == "__main__":
    unittest.main()
