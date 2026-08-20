import heapq
import importlib.util
import json
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from domain.helpers.capture_schedule import (
    build_fixed_fps_schedule,
    build_original_timing_schedule,
    build_passage_capture_plan,
)
from infra.profiling.telemetry import CaptureTimingRecorder, TelemetryContext
from mas.agents.dataset_capture_agent import DatasetCaptureBehaviour
from mas.infrastructure.frame_store import FRAME_STORE, FrameStore
from mas.utils.globals import FRAME_BUFFER
from mas_pipeline import MASStrategy
from thread_pipeline import ThreadPipeline


class FakeScheduler:
    def __init__(self):
        self.now = 0.0
        self._order = 0
        self._calls = []

    def call_later(self, delay, callback):
        deadline = self.now + delay
        self._order += 1
        heapq.heappush(self._calls, (deadline, self._order, callback))
        return SimpleNamespace(cancel=lambda: None)

    def run_next(self, lateness=0.0):
        deadline, _, callback = heapq.heappop(self._calls)
        self.now = max(self.now, deadline) + lateness
        callback()

    def run_all(self, limit=1000):
        executions = 0
        while self._calls:
            self.run_next()
            executions += 1
            if executions > limit:
                raise AssertionError("scheduler did not quiesce")


class FakeDataset:
    def __init__(self, indexes, missing=()):
        self.indexes = indexes
        self.missing = set(missing)
        self.loaded_filenames = []
        self.images = {}

    def load_index(self, tag):
        return [dict(item) for item in self.indexes[tag]]

    def load_depth(self, tag, depth_filename):
        self.loaded_filenames.append((tag, depth_filename))
        if (tag, depth_filename) in self.missing:
            return None
        return self.images.setdefault((tag, depth_filename), object())


class FakeAgent:
    def __init__(self, scheduler, frame_store, fail_visual_send=False):
        self.aid = SimpleNamespace(name="capture@localhost:5003")
        self.simulation_started = True
        self.scheduler = scheduler
        self.frame_store = frame_store
        self.sent = []
        self.frame_present_when_sent = []
        self.fail_visual_send = fail_visual_send

    def send(self, message):
        if self.fail_visual_send and message.ontology == "visual-event":
            raise RuntimeError("synthetic visual send failure")
        payload = json.loads(message.content)
        if payload.get("event_type") == "frame":
            self.frame_present_when_sent.append(
                self.frame_store.get(payload["frame_id"]) is not None
            )
        self.sent.append((self.scheduler.now, message.ontology, payload))


def make_index(times):
    return [
        {
            "relative_time_ms": float(timestamp),
            "depth_filename": f"depth-{index}.png",
            "label": "suited" if index % 2 == 0 else "background",
        }
        for index, timestamp in enumerate(times)
    ]


def pipeline_records(agent):
    return [
        (at, payload)
        for at, ontology, payload in agent.sent
        if ontology == "pipeline-event" and payload.get("event_type") is not None
    ]


def visual_records(agent):
    return [
        (at, payload)
        for at, ontology, payload in agent.sent
        if ontology == "visual-event" and payload.get("event_type") is not None
    ]


class CapturePlanParityTests(unittest.TestCase):
    def test_fixed_plan_matches_thread_helpers_and_source_filenames(self):
        times = np.array([0.0, 400.0, 1000.0])
        frames = make_index(times)

        plan = build_passage_capture_plan(
            times,
            fps=4.0,
            native_timestamps=False,
        )
        thread_schedule = build_fixed_fps_schedule(times, fps=4.0)

        self.assertEqual(list(plan.events), thread_schedule)
        self.assertEqual(
            [frames[event.source_index]["depth_filename"] for event in plan.events],
            [
                "depth-0.png",
                "depth-1.png",
                "depth-1.png",
                "depth-2.png",
                "depth-2.png",
            ],
        )

    def test_fixed_plan_uses_previous_source_on_nearest_tie(self):
        times = np.array([0.0, 1000.0])
        plan = build_passage_capture_plan(
            times,
            fps=2.0,
            native_timestamps=False,
        )

        self.assertEqual(
            [event.source_index for event in plan.events],
            [0, 0, 1],
        )

    def test_original_plan_matches_thread_native_schedule_and_cap(self):
        times = np.array([100.0, 450.0, 900.0])
        plan = build_passage_capture_plan(
            times,
            fps=None,
            native_timestamps=True,
            max_passage_seconds=0.6,
        )

        self.assertEqual(
            list(plan.events),
            build_original_timing_schedule(times, end_ms=700.0),
        )
        self.assertEqual(plan.end_timestamp_ms, 700.0)
        self.assertEqual(plan.end_offset_s, 0.6)

    def test_passage_boundary_matches_thread_pipeline(self):
        times = np.array([100.0, 450.0, 900.0])
        thread = ThreadPipeline(
            pid="test",
            mode="single",
            fps=2.0,
            max_passage_seconds=0.6,
        )
        plan = build_passage_capture_plan(
            times,
            fps=2.0,
            native_timestamps=False,
            max_passage_seconds=0.6,
        )

        self.assertEqual(plan.end_timestamp_ms, thread._passage_end_ms(times))


class PadeCaptureSchedulerTests(unittest.TestCase):
    def run_capture(
        self,
        indexes,
        *,
        fps=2.0,
        native_timestamps=False,
        max_passage_seconds=None,
        missing=(),
        telemetry_context=None,
        capture_timing_recorder=None,
        visual_agent_aid=None,
        fail_visual_send=False,
    ):
        scheduler = FakeScheduler()
        store = FrameStore()
        dataset = FakeDataset(indexes, missing=missing)
        agent = FakeAgent(
            scheduler,
            store,
            fail_visual_send=fail_visual_send,
        )
        frame_ids = iter(f"frame-{index}" for index in range(1000))
        behaviour = DatasetCaptureBehaviour(
            agent=agent,
            dataset=dataset,
            next_agent_aid="selection@localhost:5005",
            selection_agent_aid="selection@localhost:5005",
            animal_tags=list(indexes),
            fps=fps,
            max_passage_seconds=max_passage_seconds,
            native_timestamps=native_timestamps,
            frame_store=store,
            telemetry_context=telemetry_context,
            capture_timing_recorder=capture_timing_recorder,
            visual_agent_aid=visual_agent_aid,
            call_later=scheduler.call_later,
            monotonic=lambda: scheduler.now,
            iso_now=lambda: f"t={scheduler.now:.3f}",
            frame_id_factory=lambda: next(frame_ids),
        )
        behaviour.start()
        scheduler.run_all()
        return scheduler, store, dataset, agent, behaviour

    def test_capture_sets_context_and_registers_schedule_without_using_send_time(self):
        context = TelemetryContext(
            "pade-run", "pade_fixed_fps", 2.0,
            monotonic_origin_ns=0,
        )
        recorder = CaptureTimingRecorder(context)
        scheduler, _, _, _, _ = self.run_capture(
            {"N": make_index([0.0, 500.0])},
            fps=2.0,
            telemetry_context=context,
            capture_timing_recorder=recorder,
        )

        # Capture não usa Agent.send como admissão: os dois schedules ainda
        # aguardam confirmação no receptor Selection.inbox.put().
        self.assertEqual(recorder.pending_count(), 2)
        self.assertEqual(recorder.get_all_data(), [])
        self.assertEqual(scheduler.now, 0.5)

    def test_fixed_fps_end_has_no_extra_tick_for_multiple_nonmultiple_and_one_fps(self):
        cases = [
            (5.0, 2.0, 5.0),
            (5.3, 2.0, 5.3),
            (2.0, 1.0, 2.0),
        ]
        for duration_s, fps, expected_end_s in cases:
            with self.subTest(duration_s=duration_s, fps=fps):
                _, _, _, agent, _ = self.run_capture(
                    {"N": make_index([0.0, duration_s * 1000.0])},
                    fps=fps,
                )
                records = pipeline_records(agent)
                passage_end = next(
                    at for at, payload in records
                    if payload["event_type"] == "end_passage"
                )
                frame_times = [
                    payload["elapsed_time"]
                    for _, payload in records
                    if payload["event_type"] == "frame"
                ]

                self.assertAlmostEqual(passage_end, expected_end_s)
                self.assertLessEqual(frame_times[-1] / 1000.0, expected_end_s)
                self.assertNotIn(
                    (expected_end_s + 1.0 / fps) * 1000.0,
                    frame_times,
                )

    def test_late_callback_preserves_all_planned_frames_without_drift(self):
        scheduler = FakeScheduler()
        store = FrameStore()
        dataset = FakeDataset({"N": make_index([0.0, 500.0, 1000.0])})
        agent = FakeAgent(scheduler, store)
        ids = iter(("a", "b", "c"))
        behaviour = DatasetCaptureBehaviour(
            agent=agent,
            dataset=dataset,
            next_agent_aid="selection@localhost:5005",
            selection_agent_aid="selection@localhost:5005",
            animal_tags=["N"],
            fps=2.0,
            frame_store=store,
            call_later=scheduler.call_later,
            monotonic=lambda: scheduler.now,
            frame_id_factory=lambda: next(ids),
        )

        behaviour.start()
        scheduler.run_next()  # inicia a passagem e agenda o primeiro frame em t=0
        scheduler.run_next(lateness=1.2)  # callback chega depois de todos deadlines
        scheduler.run_all()

        frames = [
            payload for _, payload in pipeline_records(agent)
            if payload["event_type"] == "frame"
        ]
        end_at = next(
            at for at, payload in pipeline_records(agent)
            if payload["event_type"] == "end_passage"
        )
        self.assertEqual([item["elapsed_time"] for item in frames], [0.0, 500.0, 1000.0])
        self.assertAlmostEqual(end_at, 1.2)

    def test_original_timing_preserves_unique_timestamps_and_cap(self):
        _, _, dataset, agent, _ = self.run_capture(
            {"N": make_index([100.0, 450.0, 900.0])},
            fps=None,
            native_timestamps=True,
            max_passage_seconds=0.6,
        )
        records = pipeline_records(agent)
        frames = [
            (at, payload)
            for at, payload in records
            if payload["event_type"] == "frame"
        ]
        end_at = next(
            at for at, payload in records
            if payload["event_type"] == "end_passage"
        )

        self.assertEqual([at for at, _ in frames], [0.0, 0.35])
        self.assertEqual(
            [payload["dataset_timestamp_ms"] for _, payload in frames],
            [100.0, 450.0],
        )
        self.assertEqual(
            dataset.loaded_filenames,
            [("N", "depth-0.png"), ("N", "depth-1.png")],
        )
        self.assertAlmostEqual(end_at, 0.6)

    def test_sequence_crosses_passages_and_next_starts_without_ack(self):
        _, _, _, agent, behaviour = self.run_capture(
            {
                "N": make_index([0.0, 500.0]),
                "N+1": make_index([0.0, 500.0]),
            },
            fps=2.0,
        )
        records = pipeline_records(agent)
        summary = [
            (
                payload["stream_seq"],
                payload["event_type"],
                payload.get("passage_id"),
                payload.get("capture_index"),
                at,
            )
            for at, payload in records
        ]

        self.assertEqual([item[0] for item in summary], list(range(len(summary))))
        first_end_index = next(
            index for index, item in enumerate(summary)
            if item[1] == "end_passage" and item[2] == "N"
        )
        self.assertEqual(summary[first_end_index + 1][1:4], ("frame", "N+1", 1))
        self.assertEqual(
            summary[first_end_index][4],
            summary[first_end_index + 1][4],
        )
        self.assertEqual(summary[-1][1], "end_pipeline")
        self.assertFalse(hasattr(behaviour, "predict_agent"))

    def test_capture_emits_canonical_pipeline_events(self):
        _, _, _, agent, _ = self.run_capture(
            {"N": make_index([0.0])},
            fps=1.0,
        )

        frame_message = next(
            item for item in agent.sent if item[2].get("event_type") == "frame"
        )
        end_message = next(
            item
            for item in agent.sent
            if item[2].get("event_type") == "end_passage"
        )

        self.assertEqual(frame_message[1], "pipeline-event")
        self.assertEqual(frame_message[2]["event_type"], "frame")
        self.assertEqual(frame_message[2]["passage_id"], "N")
        self.assertEqual(frame_message[2]["capture_index"], 1)
        self.assertNotIn("animal_id", frame_message[2])
        self.assertEqual(end_message[1], "pipeline-event")
        self.assertEqual(end_message[2]["event_type"], "end_passage")
        self.assertEqual(end_message[2]["passage_id"], "N")
        self.assertEqual(end_message[2]["total_captured_frames"], 1)
        self.assertNotIn("total_frames", end_message[2])

    def test_empty_index_emits_end_and_keeps_stream_sequence(self):
        _, _, _, agent, _ = self.run_capture(
            {"empty": [], "N": make_index([0.0])},
            fps=1.0,
        )
        records = [payload for _, payload in pipeline_records(agent)]

        self.assertEqual(
            [(item["stream_seq"], item["event_type"]) for item in records],
            [(0, "end_passage"), (1, "frame"), (2, "end_passage"), (3, "end_pipeline")],
        )
        self.assertEqual(records[0]["total_captured_frames"], 0)

    def test_frame_is_stored_before_message_without_duplicate_array(self):
        _, store, dataset, agent, _ = self.run_capture(
            {"N": make_index([0.0])},
            fps=1.0,
        )
        frame_payload = next(
            payload for _, payload in pipeline_records(agent)
            if payload["event_type"] == "frame"
        )

        self.assertEqual(agent.frame_present_when_sent, [True])
        self.assertEqual(len(store), 1)
        self.assertIs(
            store.get(frame_payload["frame_id"]),
            dataset.images[("N", "depth-0.png")],
        )

    def test_missing_depth_skips_frame_but_still_emits_end(self):
        _, store, _, agent, _ = self.run_capture(
            {"N": make_index([0.0, 1000.0])},
            fps=1.0,
            missing={("N", "depth-0.png")},
        )
        records = [payload for _, payload in pipeline_records(agent)]

        self.assertEqual(
            [item["event_type"] for item in records],
            ["frame", "end_passage", "end_pipeline"],
        )
        self.assertEqual(records[0]["capture_index"], 1)
        self.assertEqual(records[1]["total_captured_frames"], 1)
        self.assertEqual(len(store), 1)

    def test_visual_disabled_preserves_exact_main_event_stream_and_has_no_leases(self):
        indexes = {
            "N": make_index([0.0, 500.0]),
            "N+1": make_index([0.0, 500.0]),
        }
        _, off_store, _, off_agent, _ = self.run_capture(indexes, fps=2.0)
        _, on_store, _, on_agent, _ = self.run_capture(
            indexes,
            fps=2.0,
            visual_agent_aid="visual@localhost:5007",
        )

        self.assertEqual(pipeline_records(off_agent), pipeline_records(on_agent))
        self.assertEqual(off_store.lease_count(owner="visual"), 0)
        self.assertEqual(on_store.lease_count(owner="visual"), 4)
        self.assertEqual(visual_records(off_agent), [])

    def test_visual_edge_has_independent_contiguous_sequence_and_no_array_payload(self):
        _, store, dataset, agent, _ = self.run_capture(
            {"N": make_index([0.0, 500.0])},
            fps=2.0,
            visual_agent_aid="visual@localhost:5007",
        )
        visual = [payload for _, payload in visual_records(agent)]

        self.assertEqual([item["stream_seq"] for item in visual], [0, 1, 2, 3])
        self.assertEqual(
            [item["event_type"] for item in visual],
            ["visual_frame", "visual_frame", "end_passage", "end_pipeline"],
        )
        for item in visual[:2]:
            self.assertNotIn("image", item)
            self.assertNotIn("raw", item)
            leased = store.read_lease(item["lease_id"], owner="visual")
            source = dataset.images[("N", item["depth_filename"])]
            self.assertIs(leased, source)

    def test_visual_send_failure_releases_leases_and_main_pipeline_continues(self):
        _, store, _, agent, _ = self.run_capture(
            {"N": make_index([0.0, 500.0])},
            fps=2.0,
            visual_agent_aid="visual@localhost:5007",
            fail_visual_send=True,
        )

        self.assertEqual(
            [payload["event_type"] for _, payload in pipeline_records(agent)],
            ["frame", "frame", "end_passage", "end_pipeline"],
        )
        self.assertEqual(store.lease_count(owner="visual"), 0)


class PadeCaptureIntegrationTests(unittest.TestCase):
    def tearDown(self):
        FRAME_STORE.clear()

    def test_legacy_frame_buffer_facade_uses_the_authoritative_store(self):
        raw = object()
        enhanced = object()

        FRAME_STORE.put("frame-1", raw)
        self.assertIs(FRAME_BUFFER.get("frame-1"), raw)

        FRAME_BUFFER["frame-1"] = enhanced
        self.assertIs(FRAME_STORE.get("frame-1"), enhanced)
        self.assertIs(FRAME_BUFFER.pop("frame-1", None), enhanced)
        self.assertNotIn("frame-1", FRAME_STORE)

    def test_mas_strategy_accepts_original_timing_without_fps(self):
        strategy = MASStrategy(
            pid="test",
            mode="single",
            fps=None,
            native_timestamps=True,
        )

        self.assertIsNone(strategy.fps)
        self.assertTrue(strategy.native_timestamps)

    def test_visual_strategy_requires_explicit_provisional_configuration(self):
        with self.assertRaises(ValueError):
            MASStrategy(
                pid="test",
                mode="single",
                fps=5.0,
                visual_event_enabled=True,
            )

        strategy = MASStrategy(
            pid="test",
            mode="single",
            fps=5.0,
            visual_event_enabled=True,
            visual_pdi_threshold=0.0875,
            visual_pixel_threshold_mm=200.0,
            visual_idle_patience=3,
        )
        self.assertTrue(strategy.visual_event_enabled)
        self.assertEqual(strategy.visual_pdi_threshold, 0.0875)
        self.assertEqual(strategy.visual_pixel_threshold_mm, 200.0)
        self.assertEqual(strategy.visual_idle_patience, 3)

    def test_entrypoint_accepts_native_timestamps_with_pade(self):
        module_path = pathlib.Path(__file__).parents[1] / "mas-main.py"
        spec = importlib.util.spec_from_file_location("mas_main_entrypoint", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        created = []

        class FakeStrategy:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.ran = False
                created.append(self)

            def run(self):
                self.ran = True

        argv = [
            "mas-main.py",
            "mas-single",
            "--native-timestamps",
            "--engine",
            "pade",
        ]
        with (
            patch.object(module, "MASStrategy", FakeStrategy),
            patch.object(module.sys, "argv", argv),
            patch.object(module.os, "makedirs"),
        ):
            module.main()

        self.assertEqual(len(created), 1)
        self.assertIsNone(created[0].kwargs["fps"])
        self.assertTrue(created[0].kwargs["native_timestamps"])
        self.assertTrue(created[0].ran)

    def test_entrypoint_passes_explicit_visual_configuration_to_pade(self):
        module_path = pathlib.Path(__file__).parents[1] / "mas-main.py"
        spec = importlib.util.spec_from_file_location(
            "mas_main_visual_entrypoint",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        created = []

        class FakeStrategy:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created.append(self)

            def run(self):
                return None

        argv = [
            "mas-main.py",
            "mas-single",
            "5",
            "--engine",
            "pade",
            "--visual-event",
            "--visual-pdi-threshold",
            "0.0875",
            "--visual-pixel-threshold-mm",
            "200.0",
            "--visual-idle-patience",
            "3",
        ]
        with (
            patch.object(module, "MASStrategy", FakeStrategy),
            patch.object(module.sys, "argv", argv),
            patch.object(module.os, "makedirs"),
        ):
            module.main()

        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].kwargs["visual_event_enabled"])
        self.assertEqual(created[0].kwargs["visual_pdi_threshold"], 0.0875)
        self.assertEqual(created[0].kwargs["visual_pixel_threshold_mm"], 200.0)
        self.assertEqual(created[0].kwargs["visual_idle_patience"], 3)


if __name__ == "__main__":
    unittest.main()
