import queue
import unittest
from unittest.mock import Mock, patch

from infra.profiling.telemetry import TelemetryContext
from mas.utils.report_collector import ReportCollector
from thread_pipeline import ThreadPipeline, _END_ANIMAL


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleep_calls = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        if seconds < 0:
            raise AssertionError(f"negative sleep requested: {seconds}")
        self.sleep_calls.append(seconds)
        self.now += seconds


class RecordingQueue:
    def __init__(self, clock):
        self.clock = clock
        self.events = []

    def put(self, item):
        self.events.append((self.clock.monotonic(), item))


class FakeDataset:
    def __init__(self, indexes):
        self.indexes = indexes
        self.loads = []

    def load_index(self, tag):
        return [dict(frame) for frame in self.indexes[tag]]

    def load_depth(self, tag, depth_filename):
        self.loads.append((tag, depth_filename))
        return object()


def make_index(timestamps_ms):
    return [
        {
            "relative_time_ms": float(timestamp),
            "depth_filename": f"frame_{timestamp}.png",
            "label": "suited",
        }
        for timestamp in timestamps_ms
    ]


class PassageTimingTests(unittest.TestCase):
    def _pipeline(self, fps, *, native=False, max_passage_seconds=None):
        pipeline = ThreadPipeline(
            pid="test",
            mode="single",
            fps=fps,
            native_timestamps=native,
            max_passage_seconds=max_passage_seconds,
        )
        pipeline._log = Mock()
        return pipeline

    def _run_capture(self, pipeline, dataset, tags):
        clock = FakeClock()
        output = RecordingQueue(clock)
        with (
            patch("thread_pipeline.time.monotonic", side_effect=clock.monotonic),
            patch("thread_pipeline.time.sleep", side_effect=clock.sleep),
        ):
            pipeline._capture_loop(dataset, tags, output)
        return clock, output.events

    @staticmethod
    def _frames(events):
        return [(timestamp, item) for timestamp, item in events
                if isinstance(item, dict)]

    @staticmethod
    def _ends(events):
        return [(timestamp, item) for timestamp, item in events
                if isinstance(item, tuple) and item and item[0] == _END_ANIMAL]

    def test_fixed_fps_multiple_duration_ends_at_real_limit(self):
        timestamps = list(range(0, 5001, 500))
        dataset = FakeDataset({"N": make_index(timestamps)})

        clock, events = self._run_capture(
            self._pipeline(2.0), dataset, ["N"]
        )

        frames = self._frames(events)
        self.assertEqual([item["elapsed_time"] for _, item in frames], timestamps)
        self.assertEqual([round(timestamp, 6) for timestamp, _ in frames],
                         [timestamp / 1000.0 for timestamp in timestamps])
        self.assertAlmostEqual(self._ends(events)[0][0], 5.0)
        self.assertAlmostEqual(clock.now, 5.0)
        self.assertEqual(len(clock.sleep_calls), 10)

    def test_fixed_fps_non_multiple_waits_only_until_real_limit(self):
        timestamps = list(range(0, 5001, 500)) + [5300]
        dataset = FakeDataset({"N": make_index(timestamps)})

        clock, events = self._run_capture(
            self._pipeline(2.0), dataset, ["N"]
        )

        frame_times = [item["elapsed_time"] for _, item in self._frames(events)]
        self.assertEqual(frame_times, [float(value) for value in range(0, 5001, 500)])
        self.assertNotIn(5500.0, frame_times)
        self.assertAlmostEqual(self._ends(events)[0][0], 5.3)
        self.assertAlmostEqual(clock.now, 5.3)
        self.assertAlmostEqual(clock.sleep_calls[-1], 0.3)

    def test_fixed_fps_one_hz_has_no_artificial_final_second(self):
        timestamps = list(range(0, 5001, 1000))
        dataset = FakeDataset({"N": make_index(timestamps)})

        clock, events = self._run_capture(
            self._pipeline(1.0), dataset, ["N"]
        )

        self.assertEqual(len(self._frames(events)), 6)
        self.assertAlmostEqual(self._ends(events)[0][0], 5.0)
        self.assertAlmostEqual(clock.now, 5.0)
        self.assertEqual(len(clock.sleep_calls), 5)

    def test_original_timing_preserves_source_timestamps_without_extra_delay(self):
        timestamps = [0, 300, 900, 1530]
        dataset = FakeDataset({"N": make_index(timestamps)})

        clock, events = self._run_capture(
            self._pipeline(None, native=True), dataset, ["N"]
        )

        frames = self._frames(events)
        self.assertEqual([item["dataset_timestamp_ms"] for _, item in frames],
                         [float(value) for value in timestamps])
        self.assertEqual([round(timestamp, 6) for timestamp, _ in frames],
                         [value / 1000.0 for value in timestamps])
        self.assertAlmostEqual(self._ends(events)[0][0], 1.53)
        self.assertAlmostEqual(clock.now, 1.53)

    def test_original_timing_cap_between_frames_ends_at_cap(self):
        timestamps = [0, 300, 900, 1530]
        dataset = FakeDataset({"N": make_index(timestamps)})

        clock, events = self._run_capture(
            self._pipeline(None, native=True, max_passage_seconds=1.2),
            dataset,
            ["N"],
        )

        frames = self._frames(events)
        self.assertEqual([item["dataset_timestamp_ms"] for _, item in frames],
                         [0.0, 300.0, 900.0])
        self.assertAlmostEqual(self._ends(events)[0][0], 1.2)
        self.assertAlmostEqual(clock.now, 1.2)

    def test_next_passage_starts_without_downstream_completion(self):
        indexes = {
            "N": make_index([0, 1000]),
            "N+1": make_index([0, 1000]),
        }
        dataset = FakeDataset(indexes)

        _, events = self._run_capture(
            self._pipeline(1.0), dataset, ["N", "N+1"]
        )

        end_n_time = next(
            timestamp for timestamp, item in self._ends(events) if item[1] == "N"
        )
        first_n1_time = next(
            timestamp for timestamp, item in self._frames(events)
            if item["animal_id"] == "N+1"
        )
        self.assertAlmostEqual(first_n1_time, end_n_time)

    def test_capture_passage_context_is_active_through_end_enqueue_only(self):
        clock = FakeClock()
        context = TelemetryContext(
            "run", "opaque-condition", 1.0, monotonic_origin_ns=0
        )

        class ContextRecordingQueue:
            def __init__(self):
                self.events = []

            def put(self, item):
                metadata = context.sample_metadata(int(clock.now * 1_000_000_000))
                self.events.append((item, metadata["capture_passage_id"]))

        output = ContextRecordingQueue()
        dataset = FakeDataset({"N": make_index([0, 1000])})
        with (
            patch("thread_pipeline.time.monotonic", side_effect=clock.monotonic),
            patch("thread_pipeline.time.sleep", side_effect=clock.sleep),
        ):
            self._pipeline(1.0)._capture_loop(
                dataset, ["N"], output, telemetry_context=context
            )

        frame_contexts = [
            passage_id for item, passage_id in output.events
            if isinstance(item, dict)
        ]
        end_context = next(
            passage_id for item, passage_id in output.events
            if isinstance(item, tuple) and item[0] == _END_ANIMAL
        )
        self.assertEqual(frame_contexts, ["N", "N"])
        self.assertEqual(end_context, "N")
        self.assertIsNone(context.sample_metadata(1_000_000_000)["capture_passage_id"])

    def test_rejected_last_frame_still_finalizes_downstream(self):
        pipeline = self._pipeline(1.0)
        pipeline._save_metrics = Mock()
        ReportCollector().reset()

        q1 = queue.Queue()
        q2 = queue.Queue()
        q3 = queue.Queue()
        q1.put({
            "frame_id": "last",
            "animal_id": "N",
            "frame_index": 1,
            "elapsed_time": 0.0,
            "label": "suited",
            "depth_filename": "last.png",
            "img": object(),
        })
        q1.put((_END_ANIMAL, "N", 1, "first", "last"))
        q1.put(None)

        selector = Mock()
        selector.evaluate_with_score.return_value = (False, 0.0)
        enhancer = Mock()
        inference = Mock()

        pipeline._select_loop(selector, q1, q2)
        pipeline._enhance_loop(enhancer, q2, q3)
        metrics = {"animals": {}}
        pipeline._predict_loop(inference, q3, 1, metrics)

        enhancer.run.assert_not_called()
        inference.predict.assert_not_called()
        pipeline._save_metrics.assert_called_once_with(metrics)
        self.assertEqual(metrics["animals"]["N"]["total_of_images"], 1)
        self.assertEqual(metrics["animals"]["N"]["suitable_images"], 0)


if __name__ == "__main__":
    unittest.main()
