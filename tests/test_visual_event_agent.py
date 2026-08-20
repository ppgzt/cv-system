import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import mas  # noqa: F401
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from twisted.internet.defer import Deferred, maybeDeferred

from domain.pipeline_events import EndPassageEvent, EndPipelineEvent, FrameEvent
from domain.pipeline_events import event_to_json
from domain.visual_activity import VisualActivityDetector, VisualState
from domain.visual_events import VisualFrameEvent, visual_event_to_json
from domain.visual_events import visual_event_from_json
from mas.agents.visual_event_agent import VisualEventAgent
from mas.agents.data_enhance_agent import DataEnhanceAgent
from mas.agents.frame_selection import FrameSelectionAgent
from mas.agents.predict_weight_agent import PredictWeightAgent
from mas.infrastructure.frame_store import FrameStore
from mas.infrastructure.ordered_inbox import OrderedInbox
from mas.utils.report_collector import ReportCollector


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


def visual_frame(store, seq, frame_id, raw, passage="N", capture_index=None):
    store.put(frame_id, raw)
    lease_id = store.retain(frame_id, owner="visual", passage_id=passage)
    return VisualFrameEvent(
        stream_seq=seq,
        lease_id=lease_id,
        passage_id=passage,
        capture_index=seq if capture_index is None else capture_index,
        elapsed_time=float(seq * 100),
        dataset_timestamp_ms=float(seq * 100),
        depth_filename=f"{frame_id}.png",
        label="suited",
    )


def end(seq, passage="N"):
    return EndPassageEvent(seq, passage, 2, "first", "last")


def acl_visual(event):
    message = ACLMessage(ACLMessage.INFORM)
    message.set_ontology("visual-event")
    message.set_sender(AID(name="capture@localhost:5003"))
    message.set_content(visual_event_to_json(event))
    return message


def acl_pipeline(event):
    message = ACLMessage(ACLMessage.INFORM)
    message.set_ontology("pipeline-event")
    message.set_sender(AID(name="upstream@localhost:5003"))
    message.set_content(event_to_json(event))
    return message


def make_agent(store, *, inbox=None, executor=maybeDeferred, publisher=None, reports_dir=None):
    agent = VisualEventAgent(
        aid=AID(name="visual@localhost:5007"),
        capture_agent_aid="capture@localhost:5003",
        pid="visual-test",
        pdi_threshold=0.0875,
        idle_patience_frames=2,
        pixel_threshold_mm=200.0,
        frame_store=store,
        inbox=inbox,
        state_publisher=publisher,
        defer_executor=executor,
        reports_dir=reports_dir or tempfile.gettempdir(),
    )
    agent.sent_messages = []
    agent.send = agent.sent_messages.append
    return agent


class VisualEventAgentTests(unittest.TestCase):
    def test_visual_contract_round_trip_contains_metadata_but_no_array(self):
        store = FrameStore()
        event = visual_frame(
            store, 0, "A", np.full((240, 320), 1500, dtype=np.uint16), capture_index=1
        )

        payload = visual_event_to_json(event)

        self.assertEqual(visual_event_from_json(payload), event)
        self.assertNotIn("array", payload)
        self.assertNotIn("image", payload)
        self.assertNotIn("raw", payload)

    def test_physical_out_of_order_end_waits_for_both_frames(self):
        store = FrameStore()
        executor = ManualExecutor()
        agent = make_agent(
            store,
            inbox=OrderedInbox(expected_seq=10),
            executor=executor,
        )
        first_raw = np.full((240, 320), 1500, dtype=np.uint16)
        second_raw = np.full((240, 320), 1500, dtype=np.uint16)
        second_raw[100:150, 100:150] = 2000  # Variação de 500mm na ROI -> ACTIVE

        first = visual_frame(
            store, 10, "A", first_raw, capture_index=1
        )
        second = visual_frame(
            store, 11, "B", second_raw, capture_index=2
        )

        agent.react(acl_visual(first))
        agent.react(acl_visual(end(12)))
        agent.react(acl_visual(second))

        self.assertTrue(agent._processing)
        self.assertEqual(agent.inbox.qsize(), 2)
        executor.complete_next()
        self.assertEqual([item.capture_index for item in agent.observations], [1])
        self.assertTrue(agent._processing)
        executor.complete_next()

        self.assertEqual([item.capture_index for item in agent.observations], [1, 2])
        self.assertIs(agent.observations[1].visual_state, VisualState.ACTIVE)
        self.assertEqual(agent.passage_final_states["N"], "ACTIVE")
        self.assertIsNone(agent.detector.previous_raw)
        self.assertFalse(agent._processing)
        self.assertEqual(store.lease_count(owner="visual"), 0)

    def test_selection_discard_or_enhance_overwrite_cannot_invalidate_raw_lease(self):
        for mutation in ("discard", "overwrite"):
            with self.subTest(mutation=mutation):
                store = FrameStore()
                raw = np.arange(4, dtype=np.uint16).reshape(2, 2)
                event = visual_frame(store, 0, "A", raw, capture_index=1)
                if mutation == "discard":
                    store.discard("A")
                else:
                    enhanced = np.ones((2, 2, 3), dtype=np.float32)
                    store.put("A", enhanced)

                agent = make_agent(store)
                agent.react(acl_visual(event))

                self.assertEqual(len(agent.observations), 1)
                self.assertIsNone(agent.observations[0].mad)
                self.assertTrue(np.shares_memory(raw, agent.detector.previous_raw))
                self.assertEqual(store.lease_count(owner="visual"), 0)

    def test_result_publication_failure_releases_and_allows_end(self):
        store = FrameStore()
        event = visual_frame(
            store, 0, "A", np.zeros((2, 2), dtype=np.uint16), capture_index=1
        )

        def fail_publish(_event):
            raise RuntimeError("synthetic publisher failure")

        agent = make_agent(store, publisher=fail_publish)
        agent.react(acl_visual(event))
        agent.react(acl_visual(end(1)))

        self.assertFalse(agent._processing)
        self.assertEqual(store.lease_count(owner="visual"), 0)
        self.assertIn("N", agent.passage_final_states)

    def test_detector_error_releases_lease_and_next_passage_continues(self):
        store = FrameStore()
        agent = make_agent(store)
        event = visual_frame(
            store, 0, "A", np.zeros((2, 2), dtype=np.uint16), capture_index=1
        )
        original = agent.detector.observe
        calls = 0

        def fail_once(raw):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("synthetic MAD failure")
            return original(raw)

        agent.detector.observe = fail_once
        agent.react(acl_visual(event))
        agent.react(acl_visual(end(1)))
        next_event = visual_frame(
            store,
            2,
            "B",
            np.ones((2, 2), dtype=np.uint16),
            passage="N+1",
            capture_index=1,
        )
        agent.react(acl_visual(next_event))
        agent.react(acl_visual(end(3, "N+1")))

        self.assertFalse(agent._processing)
        self.assertEqual([item.passage_id for item in agent.observations], ["N+1"])
        self.assertEqual(store.lease_count(owner="visual"), 0)
        self.assertEqual(set(agent.passage_final_states), {"N", "N+1"})

    def test_end_pipeline_persists_closes_inbox_and_never_requests_shutdown(self):
        store = FrameStore()
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(store, reports_dir=tmp)
            event = visual_frame(
                store, 0, "A", np.zeros((2, 2), dtype=np.uint16), capture_index=1
            )
            with patch("twisted.internet.reactor.stop") as reactor_stop:
                agent.react(acl_visual(event))
                agent.react(acl_visual(end(1)))
                agent.react(acl_visual(EndPipelineEvent(2)))

            self.assertTrue(agent.inbox.closed)
            self.assertFalse(agent._processing)
            output = Path(tmp) / "visual-test" / "visual_activity.csv"
            self.assertTrue(output.is_file())
            lines = output.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            reactor_stop.assert_not_called()

    def test_on_start_only_sends_readiness_metadata(self):
        agent = make_agent(FrameStore())
        agent.on_start()

        self.assertEqual(len(agent.sent_messages), 1)
        message = agent.sent_messages[0]
        self.assertEqual(message.ontology, "agent-ready")
        self.assertEqual(json.loads(message.content)["agent"], agent.aid.name)

    def test_main_data_plane_is_identical_with_visual_off_or_on(self):
        class SelectionAdapter:
            def evaluate_with_score(self, _elapsed, raw):
                accepted = int(raw[0, 0]) == 10
                return accepted, 0.9 if accepted else 0.1

        class EnhanceAdapter:
            def run(self, raw):
                return raw.astype(np.float32) * 2.0

        class InferenceAdapter:
            def load_model(self):
                return None

            def predict(self, images):
                return [[float(np.mean(image))] for image in images]

        def run(visual_enabled, visual_first):
            ReportCollector().reset()
            store = FrameStore()
            prediction = PredictWeightAgent(
                aid=AID(name="predict@localhost:5006"),
                inference_adapter=InferenceAdapter(),
                mode="single",
                pid="visual-noninterference",
                frame_store=store,
                defer_executor=maybeDeferred,
                call_later=lambda *_args: None,
                shutdown_callback=lambda: None,
            )
            prediction._save_metrics = lambda: None
            prediction.send = lambda _message: None
            enhance = DataEnhanceAgent(
                aid=AID(name="enhance@localhost:5004"),
                data_enhance_adapter=EnhanceAdapter(),
                next_agent_aid=prediction.aid.name,
                frame_store=store,
                defer_executor=maybeDeferred,
            )
            selection = FrameSelectionAgent(
                aid=AID(name="selection@localhost:5005"),
                frame_selection_adapter=SelectionAdapter(),
                next_agent_aid=enhance.aid.name,
                frame_store=store,
                defer_executor=maybeDeferred,
            )
            selection._record_selection = lambda *_args: None
            def forward(message, sender, receiver):
                if message.ontology == "pipeline-event":
                    message.set_sender(sender)
                    receiver.react(message)

            selection.send = lambda message: forward(
                message, selection.aid, enhance
            )
            enhance.send = lambda message: forward(
                message, enhance.aid, prediction
            )
            visual = make_agent(store) if visual_enabled else None

            raws = [
                np.full((2, 2), 10, dtype=np.uint16),
                np.full((2, 2), 50, dtype=np.uint16),
            ]
            for index, raw in enumerate(raws):
                frame_id = f"F{index}"
                store.put(frame_id, raw)
                main_event = FrameEvent(
                    stream_seq=index,
                    frame_id=frame_id,
                    passage_id="N",
                    capture_index=index + 1,
                    elapsed_time=float(index * 100),
                    depth_filename=f"{frame_id}.png",
                    label="suited" if index == 0 else "background",
                    dataset_timestamp_ms=float(index * 100),
                )
                if visual is not None:
                    lease_id = store.retain(
                        frame_id, owner="visual", passage_id="N"
                    )
                    visual_event = VisualFrameEvent(
                        stream_seq=index,
                        lease_id=lease_id,
                        passage_id="N",
                        capture_index=index + 1,
                        elapsed_time=float(index * 100),
                        dataset_timestamp_ms=float(index * 100),
                        depth_filename=f"{frame_id}.png",
                        label=main_event.label,
                    )
                    if visual_first:
                        visual.react(acl_visual(visual_event))
                        selection.react(acl_pipeline(main_event))
                    else:
                        selection.react(acl_pipeline(main_event))
                        visual.react(acl_visual(visual_event))
                else:
                    selection.react(acl_pipeline(main_event))

            main_end = EndPassageEvent(2, "N", 2, "first", "last")
            selection.react(acl_pipeline(main_end))
            selection.react(acl_pipeline(EndPipelineEvent(3)))
            if visual is not None:
                visual.react(acl_visual(EndPassageEvent(2, "N", 2, "first", "last")))
                visual.react(acl_visual(EndPipelineEvent(3)))

            report = ReportCollector()
            return {
                "forwarded": selection.forwarded,
                "discarded": selection.discarded,
                "predictions": list(report.prediction_data.get("N", [])),
                "final": report.final_predictions.get("N"),
                "store_size": len(store),
                "visual_states": (
                    [event.visual_state.value for event in visual.observations]
                    if visual is not None else None
                ),
            }

        off = run(False, False)
        on_main_first = run(True, False)
        on_visual_first = run(True, True)
        functional_keys = ("forwarded", "discarded", "predictions", "final", "store_size")
        self.assertEqual(
            {key: off[key] for key in functional_keys},
            {key: on_main_first[key] for key in functional_keys},
        )
        self.assertEqual(
            {key: off[key] for key in functional_keys},
            {key: on_visual_first[key] for key in functional_keys},
        )
        self.assertEqual(on_main_first["visual_states"], on_visual_first["visual_states"])


if __name__ == "__main__":
    unittest.main()
