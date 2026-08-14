import queue
import unittest
from unittest.mock import patch

import mas  # noqa: F401  (expoe o PADE vendorizado no sys.path)

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from twisted.internet.defer import Deferred, maybeDeferred

from domain.pipeline_events import (
    EndPassageEvent,
    EndPipelineEvent,
    FrameEvent,
    event_to_json,
)
from mas.agents.data_enhance_agent import DataEnhanceAgent
from mas.agents.frame_selection import FrameSelectionAgent
from mas.agents.predict_weight_agent import PredictWeightAgent
from mas.infrastructure.frame_store import FrameStore
from mas.infrastructure.ordered_inbox import OrderedInbox
from mas.utils.report_collector import ReportCollector
from thread_pipeline import ThreadPipeline, _END_ANIMAL


def frame(seq, frame_id, passage_id="N", capture_index=None):
    return FrameEvent(
        stream_seq=seq,
        frame_id=frame_id,
        passage_id=passage_id,
        capture_index=seq if capture_index is None else capture_index,
        elapsed_time=float(seq * 100),
        depth_filename=f"{frame_id}.png",
        label="suited",
        dataset_timestamp_ms=None,
    )


def end(seq, passage_id="N", total=2):
    return EndPassageEvent(
        stream_seq=seq,
        passage_id=passage_id,
        total_captured_frames=total,
        first_capture_time=f"first-{passage_id}",
        last_capture_time=f"last-{passage_id}",
    )


def acl_event(event):
    message = ACLMessage(ACLMessage.INFORM)
    message.set_ontology("pipeline-event")
    message.set_sender(AID(name="upstream@localhost:5004"))
    message.set_content(event_to_json(event))
    return message


class ManualExecutor:
    def __init__(self):
        self.pending = []

    def __call__(self, function, *args):
        deferred = Deferred()
        self.pending.append((deferred, function, args))
        return deferred

    def complete_next(self):
        deferred, function, args = self.pending.pop(0)
        try:
            deferred.callback(function(*args))
        except Exception as exc:
            deferred.errback(exc)


class FakeInferenceAdapter:
    def __init__(self, weights=None, errors=()):
        self.weights = weights or {}
        self.errors = set(errors)
        self.calls = []

    def load_model(self):
        return None

    def predict(self, images):
        self.calls.append(list(images))
        if any(image in self.errors for image in images):
            raise RuntimeError("synthetic prediction error")
        return [[self.weights[image]] for image in images]


def make_predictor(
    store,
    adapter,
    *,
    mode="single",
    inbox=None,
    executor=maybeDeferred,
):
    scheduled_shutdown = []
    agent = PredictWeightAgent(
        aid=AID(name="predict@localhost:5006"),
        inference_adapter=adapter,
        mode=mode,
        pid="test-pade-prediction",
        frame_store=store,
        inbox=inbox,
        defer_executor=executor,
        call_later=lambda delay, callback: scheduled_shutdown.append(
            (delay, callback)
        ),
        shutdown_callback=lambda: None,
        now=iter(f"t-{index}" for index in range(1000)).__next__,
    )
    agent._save_metrics = lambda: None
    agent.sent_messages = []
    agent.send = agent.sent_messages.append
    agent.scheduled_shutdown = scheduled_shutdown
    return agent


class PredictionOrderingTests(unittest.TestCase):
    def setUp(self):
        ReportCollector().reset()

    def test_physical_10_12_11_and_slow_frame_cannot_let_end_overtake(self):
        store = FrameStore()
        store.put("A", "enhanced-A")
        store.put("B", "enhanced-B")
        executor = ManualExecutor()
        predictor = make_predictor(
            store,
            FakeInferenceAdapter({"enhanced-A": 10.0, "enhanced-B": 14.0}),
            inbox=OrderedInbox(expected_seq=10),
            executor=executor,
        )

        predictor.react(acl_event(frame(10, "A", capture_index=1)))
        predictor.react(acl_event(end(12)))
        predictor.react(acl_event(frame(11, "B", capture_index=2)))

        self.assertTrue(predictor._processing)
        self.assertNotIn("N", predictor._finalized)
        self.assertEqual(predictor.inbox.qsize(), 2)

        executor.complete_next()
        self.assertNotIn("N", predictor._finalized)
        self.assertEqual(len(executor.pending), 1)
        executor.complete_next()

        self.assertIn("N", predictor._finalized)
        self.assertEqual(
            ReportCollector().final_predictions["N"],
            12.0,
        )
        self.assertFalse(predictor._processing)
        self.assertEqual(len(store), 0)

    def test_single_error_does_not_block_following_frame_or_end(self):
        store = FrameStore()
        for frame_id, value in (
            ("A", "enhanced-A"),
            ("B", "prediction-error"),
            ("C", "enhanced-C"),
        ):
            store.put(frame_id, value)
        adapter = FakeInferenceAdapter(
            {"enhanced-A": 10.0, "enhanced-C": 14.0},
            errors={"prediction-error"},
        )
        predictor = make_predictor(store, adapter)

        for event in (
            frame(0, "A", capture_index=1),
            frame(1, "B", capture_index=2),
            frame(2, "C", capture_index=3),
            end(3, total=3),
        ):
            predictor.react(acl_event(event))

        self.assertEqual(ReportCollector().final_predictions["N"], 12.0)
        self.assertEqual(predictor.metrics["animals"]["N"]["suitable_images"], 2)
        self.assertFalse(predictor._processing)
        self.assertEqual(len(store), 0)

    def test_callback_exception_releases_exactly_once_and_end_still_finalizes(self):
        store = FrameStore()
        store.put("A", "enhanced-A")
        predictor = make_predictor(
            store,
            FakeInferenceAdapter({"enhanced-A": 9.0}),
        )
        finish_calls = []
        original_finish = predictor._finish_current_event

        def tracked_finish(event, **kwargs):
            finish_calls.append(event.stream_seq)
            original_finish(event, **kwargs)

        predictor._finish_current_event = tracked_finish
        predictor._record_single_metric = lambda *_args: (_ for _ in ()).throw(
            RuntimeError("synthetic metric error")
        )

        predictor.react(acl_event(frame(0, "A")))
        predictor.react(acl_event(end(1, total=1)))

        self.assertEqual(finish_calls.count(0), 1)
        self.assertEqual(finish_calls.count(1), 1)
        self.assertIn("N", predictor._finalized)
        self.assertFalse(predictor._processing)

    def test_zero_accepted_and_consecutive_passages_are_independent(self):
        store = FrameStore()
        store.put("B", "enhanced-B")
        predictor = make_predictor(
            store,
            FakeInferenceAdapter({"enhanced-B": 21.0}),
        )

        for event in (
            end(0, "empty", 3),
            frame(1, "B", "next", 1),
            end(2, "next", 1),
        ):
            predictor.react(acl_event(event))

        self.assertEqual(ReportCollector().final_predictions["empty"], 0.0)
        self.assertEqual(ReportCollector().final_predictions["next"], 21.0)
        self.assertEqual(
            predictor.metrics["animals"]["empty"]["suitable_images"],
            0,
        )

    def test_end_pipeline_runs_only_after_prior_end_and_schedules_shutdown_once(self):
        store = FrameStore()
        store.put("A", "enhanced-A")
        executor = ManualExecutor()
        predictor = make_predictor(
            store,
            FakeInferenceAdapter({"enhanced-A": 7.0}),
            executor=executor,
        )
        saved = []
        predictor._save_metrics = lambda: saved.append("saved")

        for event in (frame(0, "A"), end(1, total=1), EndPipelineEvent(2)):
            predictor.react(acl_event(event))

        self.assertEqual(saved, [])
        self.assertFalse(predictor.inbox.closed)
        executor.complete_next()

        self.assertEqual(saved, ["saved"])
        self.assertTrue(predictor.inbox.closed)
        self.assertTrue(predictor._global_finished)
        self.assertEqual(len(predictor.scheduled_shutdown), 1)
        self.assertEqual(predictor.scheduled_shutdown[0][0], 1.0)

    def test_legacy_finalization_protocol_is_not_part_of_predictor(self):
        predictor = make_predictor(FrameStore(), FakeInferenceAdapter())
        self.assertFalse(hasattr(predictor, "notify_capture_done"))
        self.assertFalse(hasattr(predictor, "_maybe_finalize"))
        self.assertFalse(hasattr(predictor, "_capture_done"))
        self.assertFalse(hasattr(predictor, "_in_flight"))
        self.assertFalse(hasattr(predictor, "expected_counts"))


class PredictionBatchTests(unittest.TestCase):
    def setUp(self):
        ReportCollector().reset()

    def test_end_triggers_batch_and_aggregates_mean(self):
        store = FrameStore()
        for frame_id, value in (("A", "enhanced-A"), ("B", "enhanced-B")):
            store.put(frame_id, value)
        adapter = FakeInferenceAdapter({"enhanced-A": 10.0, "enhanced-B": 14.0})
        predictor = make_predictor(store, adapter, mode="batch")

        for event in (frame(0, "A", capture_index=1), frame(1, "B", capture_index=2)):
            predictor.react(acl_event(event))
        self.assertEqual(adapter.calls, [])
        predictor.react(acl_event(end(2, total=5)))

        self.assertEqual(adapter.calls, [["enhanced-A", "enhanced-B"]])
        self.assertEqual(ReportCollector().final_predictions["N"], 12.0)
        self.assertEqual(predictor.metrics["animals"]["N"]["suitable_images"], 2)
        self.assertIn("5", predictor.metrics["animals"]["N"]["imgs"])
        self.assertEqual(len(store), 0)

    def test_empty_batch_and_batch_error_finalize_as_zero_and_continue(self):
        store = FrameStore()
        store.put("bad", "prediction-error")
        store.put("good", "enhanced-good")
        adapter = FakeInferenceAdapter(
            {"enhanced-good": 18.0},
            errors={"prediction-error"},
        )
        predictor = make_predictor(store, adapter, mode="batch")

        for event in (
            end(0, "empty", 0),
            frame(1, "bad", "failed", 1),
            end(2, "failed", 1),
            frame(3, "good", "next", 1),
            end(4, "next", 1),
        ):
            predictor.react(acl_event(event))

        self.assertEqual(ReportCollector().final_predictions["empty"], 0.0)
        self.assertEqual(ReportCollector().final_predictions["failed"], 0.0)
        self.assertEqual(ReportCollector().final_predictions["next"], 18.0)
        self.assertFalse(predictor._processing)


class FakeSelectionAdapter:
    def load_model(self):
        return None

    def evaluate_with_score(self, _elapsed, raw):
        return raw != "reject", 0.9 if raw != "reject" else 0.1


class FakeEnhanceAdapter:
    def run(self, raw):
        return f"enhanced:{raw}"


class OrderedDataPlaneIntegrationTests(unittest.TestCase):
    def setUp(self):
        ReportCollector().reset()

    def test_canonical_data_plane_handles_rejection_empty_passage_and_end_pipeline(self):
        store = FrameStore()
        store.put("A", "accept")
        store.put("B", "reject")
        adapter = FakeInferenceAdapter({"enhanced:accept": 13.0})
        predictor = make_predictor(store, adapter)
        saved = []
        predictor._save_metrics = lambda: saved.append(True)

        enhance = DataEnhanceAgent(
            aid=AID(name="enhance@localhost:5004"),
            data_enhance_adapter=FakeEnhanceAdapter(),
            next_agent_aid=predictor.aid.name,
            frame_store=store,
            defer_executor=maybeDeferred,
        )
        selection = FrameSelectionAgent(
            aid=AID(name="selection@localhost:5005"),
            frame_selection_adapter=FakeSelectionAdapter(),
            next_agent_aid=enhance.aid.name,
            frame_store=store,
            defer_executor=maybeDeferred,
        )

        def forward(target, sender):
            def send(message):
                if message.ontology != "pipeline-event":
                    return
                message.set_sender(sender)
                target.react(message)
            return send

        selection._record_selection = lambda *_args: None
        selection.send = forward(enhance, selection.aid)
        enhance.send = forward(predictor, enhance.aid)

        for event in (
            frame(0, "A", "N", 1),
            frame(1, "B", "N", 2),
            end(2, "N", 2),
            end(3, "empty", 0),
            EndPipelineEvent(4),
        ):
            selection.react(acl_event(event))

        self.assertEqual(ReportCollector().final_predictions["N"], 13.0)
        self.assertEqual(ReportCollector().final_predictions["empty"], 0.0)
        self.assertEqual(saved, [True])
        self.assertEqual(len(store), 0)
        self.assertTrue(selection.inbox.closed)
        self.assertTrue(enhance.inbox.closed)
        self.assertTrue(predictor.inbox.closed)

    def test_prediction_matches_thread_pipeline_for_same_single_fixture(self):
        adapter = FakeInferenceAdapter({"enhanced-A": 10.0, "enhanced-C": 14.0})
        thread_metrics = {"animals": {}}
        q3 = queue.Queue()
        for index, frame_id in enumerate(("A", "C"), start=1):
            q3.put({
                "frame_id": frame_id,
                "animal_id": "N",
                "frame_index": index,
                "depth_filename": f"{frame_id}.png",
                "label": "suited",
                "img": f"enhanced-{frame_id}",
            })
        q3.put((_END_ANIMAL, "N", 3, "first-N", "last-N"))
        q3.put(None)
        thread = ThreadPipeline(pid="test", mode="single", fps=1.0)
        thread._save_metrics = lambda _metrics: None
        thread._predict_loop(adapter, q3, 1, thread_metrics)
        thread_final = ReportCollector().final_predictions["N"]

        ReportCollector().reset()
        store = FrameStore()
        store.put("A", "enhanced-A")
        store.put("C", "enhanced-C")
        predictor = make_predictor(store, adapter)
        for event in (
            frame(0, "A", capture_index=1),
            frame(1, "C", capture_index=2),
            end(2, total=3),
        ):
            predictor.react(acl_event(event))

        pade_entry = predictor.metrics["animals"]["N"]
        thread_entry = thread_metrics["animals"]["N"]
        self.assertEqual(ReportCollector().final_predictions["N"], thread_final)
        self.assertEqual(pade_entry["total_of_images"], thread_entry["total_of_images"])
        self.assertEqual(pade_entry["suitable_images"], thread_entry["suitable_images"])
        self.assertEqual(set(pade_entry["imgs"]), set(thread_entry["imgs"]))

    def test_prediction_matches_thread_pipeline_for_same_batch_fixture(self):
        thread_adapter = FakeInferenceAdapter(
            {"enhanced-A": 8.0, "enhanced-C": 16.0}
        )
        thread_metrics = {"animals": {}}
        q3 = queue.Queue()
        for index, frame_id in enumerate(("A", "C"), start=1):
            q3.put({
                "frame_id": frame_id,
                "animal_id": "N",
                "frame_index": index,
                "depth_filename": f"{frame_id}.png",
                "label": "suited",
                "img": f"enhanced-{frame_id}",
            })
        q3.put((_END_ANIMAL, "N", 4, "first-N", "last-N"))
        q3.put(None)
        thread = ThreadPipeline(pid="test", mode="batch", fps=1.0)
        thread._save_metrics = lambda _metrics: None
        thread._predict_loop(thread_adapter, q3, 1, thread_metrics)
        thread_final = ReportCollector().final_predictions["N"]

        ReportCollector().reset()
        store = FrameStore()
        store.put("A", "enhanced-A")
        store.put("C", "enhanced-C")
        pade_adapter = FakeInferenceAdapter(
            {"enhanced-A": 8.0, "enhanced-C": 16.0}
        )
        predictor = make_predictor(store, pade_adapter, mode="batch")
        for event in (
            frame(0, "A", capture_index=1),
            frame(1, "C", capture_index=2),
            end(2, total=4),
        ):
            predictor.react(acl_event(event))

        pade_entry = predictor.metrics["animals"]["N"]
        thread_entry = thread_metrics["animals"]["N"]
        self.assertEqual(ReportCollector().final_predictions["N"], thread_final)
        self.assertEqual(pade_adapter.calls, thread_adapter.calls)
        self.assertEqual(pade_entry["total_of_images"], thread_entry["total_of_images"])
        self.assertEqual(pade_entry["suitable_images"], thread_entry["suitable_images"])
        self.assertEqual(set(pade_entry["imgs"]), set(thread_entry["imgs"]))


if __name__ == "__main__":
    unittest.main()
