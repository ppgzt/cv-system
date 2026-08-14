import json
import queue
import unittest
from unittest.mock import patch

import mas  # noqa: F401  (expõe o PADE vendorizado no sys.path)

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from twisted.internet.defer import Deferred, maybeDeferred

from domain.pipeline_events import (
    EndPassageEvent,
    EndPipelineEvent,
    FrameEvent,
    event_from_json,
    event_to_json,
)
from mas.agents.data_enhance_agent import DataEnhanceAgent
from mas.agents.frame_selection import FrameSelectionAgent
from mas.infrastructure.frame_store import FrameStore
from mas.infrastructure.ordered_inbox import OrderedInbox
from mas.infrastructure.stream_sequence import StreamSequencer
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
    message.set_sender(AID(name="upstream@localhost:5003"))
    message.set_content(event_to_json(event))
    return message


def pipeline_events(messages):
    return [
        event_from_json(message.content)
        for message in messages
        if message.ontology == "pipeline-event"
    ]


class ManualExecutor:
    """Executor deterministico que mantem o callback artificialmente lento."""

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


class FakeSelectionAdapter:
    def __init__(self, decisions):
        self.decisions = decisions

    def evaluate_with_score(self, _elapsed, raw):
        decision = self.decisions[raw]
        if isinstance(decision, Exception):
            raise decision
        return bool(decision), 0.9 if decision else 0.1


class FakeEnhanceAdapter:
    def run(self, raw):
        if raw == "enhance-error":
            raise RuntimeError("synthetic enhance error")
        return f"enhanced:{raw}"


def make_selection(
    store,
    decisions,
    *,
    inbox=None,
    sequencer=None,
    executor=maybeDeferred,
):
    agent = FrameSelectionAgent(
        aid=AID(name="selection@localhost:5005"),
        frame_selection_adapter=FakeSelectionAdapter(decisions),
        next_agent_aid="enhance@localhost:5004",
        frame_store=store,
        inbox=inbox,
        output_sequencer=sequencer,
        defer_executor=executor,
    )
    agent._record_selection = lambda *_args: None
    agent.sent_messages = []
    agent.send = agent.sent_messages.append
    return agent


def make_enhance(
    store,
    *,
    inbox=None,
    sequencer=None,
    executor=maybeDeferred,
):
    agent = DataEnhanceAgent(
        aid=AID(name="enhance@localhost:5004"),
        data_enhance_adapter=FakeEnhanceAdapter(),
        next_agent_aid="predict@localhost:5006",
        frame_store=store,
        inbox=inbox,
        output_sequencer=sequencer,
        defer_executor=executor,
    )
    agent.sent_messages = []
    agent.send = agent.sent_messages.append
    return agent


class SelectionOrderedStageTests(unittest.TestCase):
    def test_physical_10_12_11_is_processed_as_10_11_12(self):
        store = FrameStore()
        store.put("A", "raw-A")
        store.put("B", "raw-B")
        executor = ManualExecutor()
        selection = make_selection(
            store,
            {"raw-A": True, "raw-B": True},
            inbox=OrderedInbox(expected_seq=10),
            executor=executor,
        )

        selection.react(acl_event(frame(10, "A", capture_index=1)))
        selection.react(acl_event(end(12)))
        selection.react(acl_event(frame(11, "B", capture_index=2)))

        self.assertEqual(selection.inbox.qsize(), 2)
        self.assertEqual(selection.inbox.ready_qsize(), 2)
        self.assertEqual(pipeline_events(selection.sent_messages), [])

        executor.complete_next()
        self.assertEqual(
            [event.frame_id for event in pipeline_events(selection.sent_messages)],
            ["A"],
        )
        executor.complete_next()

        observed = pipeline_events(selection.sent_messages)
        self.assertEqual(
            [(type(event).__name__, getattr(event, "frame_id", None)) for event in observed],
            [("FrameEvent", "A"), ("FrameEvent", "B"), ("EndPassageEvent", None)],
        )
        self.assertEqual([event.stream_seq for event in observed], [0, 1, 2])

    def test_rejection_removes_frame_but_end_keeps_its_position(self):
        store = FrameStore()
        store.put("A", "raw-A")
        store.put("B", "raw-B")
        selection = make_selection(
            store,
            {"raw-A": True, "raw-B": False},
        )

        for event in (frame(0, "A"), frame(1, "B"), end(2)):
            selection.react(acl_event(event))

        observed = pipeline_events(selection.sent_messages)
        self.assertEqual(
            [(type(item).__name__, getattr(item, "frame_id", None)) for item in observed],
            [("FrameEvent", "A"), ("EndPassageEvent", None)],
        )
        self.assertIsNotNone(store.get("A"))
        self.assertIsNone(store.get("B"))
        self.assertFalse(hasattr(selection, "expected_frames"))
        self.assertFalse(hasattr(selection, "processed_frames"))

    def test_selection_error_discards_only_that_frame_and_does_not_block_end(self):
        store = FrameStore()
        store.put("A", "raw-A")
        store.put("B", "selection-error")
        store.put("C", "raw-C")
        selection = make_selection(
            store,
            {
                "raw-A": True,
                "selection-error": RuntimeError("synthetic selection error"),
                "raw-C": True,
            },
        )

        events = (frame(0, "A"), frame(1, "B"), frame(2, "C"), end(3, total=3))
        for event in events:
            selection.react(acl_event(event))

        observed = pipeline_events(selection.sent_messages)
        self.assertEqual(
            [getattr(item, "frame_id", "END") for item in observed],
            ["A", "C", "END"],
        )
        self.assertIsNone(store.get("B"))

    def test_consecutive_passages_and_end_pipeline_are_resequenced(self):
        store = FrameStore()
        store.put("N-A", "raw-N")
        store.put("N1-A", "raw-N1")
        selection = make_selection(
            store,
            {"raw-N": True, "raw-N1": True},
        )

        events = (
            frame(0, "N-A", "N", 1),
            end(1, "N", 1),
            frame(2, "N1-A", "N+1", 1),
            end(3, "N+1", 1),
            EndPipelineEvent(4),
        )
        for event in events:
            selection.react(acl_event(event))

        observed = pipeline_events(selection.sent_messages)
        self.assertEqual(
            [(type(item).__name__, getattr(item, "passage_id", None)) for item in observed],
            [
                ("FrameEvent", "N"),
                ("EndPassageEvent", "N"),
                ("FrameEvent", "N+1"),
                ("EndPassageEvent", "N+1"),
                ("EndPipelineEvent", None),
            ],
        )
        self.assertEqual([item.stream_seq for item in observed], list(range(5)))
        self.assertTrue(selection.inbox.closed)

    def test_send_failure_releases_once_and_allows_following_frame_and_end(self):
        store = FrameStore()
        store.put("A", "raw-A")
        store.put("C", "raw-C")
        selection = make_selection(
            store,
            {"raw-A": True, "raw-C": True},
        )
        finish_calls = []
        original_finish = selection._finish_current_frame

        def tracked_finish(event):
            finish_calls.append(event.frame_id)
            original_finish(event)

        selection._finish_current_frame = tracked_finish
        sent = []

        def fail_first_frame_send(message):
            if message.ontology != "pipeline-event":
                sent.append(message)
                return
            event = event_from_json(message.content)
            if isinstance(event, FrameEvent) and event.frame_id == "A":
                raise RuntimeError("synthetic send error")
            sent.append(message)

        selection.send = fail_first_frame_send
        for event in (frame(0, "A"), frame(1, "C"), end(2)):
            selection.react(acl_event(event))

        self.assertEqual(
            [getattr(item, "frame_id", "END") for item in pipeline_events(sent)],
            ["C", "END"],
        )
        self.assertEqual(finish_calls, ["A", "C"])
        self.assertFalse(selection._processing)
        self.assertIsNone(selection._active_frame_seq)

    def test_errback_cleanup_failure_still_releases_once_and_reaches_end(self):
        store = FrameStore()
        store.put("B", "selection-error")
        store.put("C", "raw-C")
        original_discard = store.discard

        def failing_discard(frame_id):
            if frame_id == "B":
                raise RuntimeError("synthetic cleanup error")
            return original_discard(frame_id)

        store.discard = failing_discard
        selection = make_selection(
            store,
            {
                "selection-error": RuntimeError("synthetic adapter error"),
                "raw-C": True,
            },
        )
        finish_calls = []
        original_finish = selection._finish_current_frame

        def tracked_finish(event):
            finish_calls.append(event.frame_id)
            original_finish(event)

        selection._finish_current_frame = tracked_finish
        for event in (frame(0, "B"), frame(1, "C"), end(2)):
            selection.react(acl_event(event))

        self.assertEqual(
            [getattr(item, "frame_id", "END") for item in pipeline_events(selection.sent_messages)],
            ["C", "END"],
        )
        self.assertEqual(finish_calls, ["B", "C"])
        self.assertFalse(selection._processing)


class EnhanceOrderedStageTests(unittest.TestCase):
    def test_slow_transform_and_early_end_cannot_overtake_frames(self):
        store = FrameStore()
        store.put("A", "raw-A")
        store.put("B", "raw-B")
        executor = ManualExecutor()
        enhance = make_enhance(store, executor=executor)

        enhance.react(acl_event(frame(0, "A", capture_index=1)))
        enhance.react(acl_event(end(2)))
        enhance.react(acl_event(frame(1, "B", capture_index=2)))

        self.assertEqual(enhance.inbox.qsize(), 2)
        self.assertEqual(pipeline_events(enhance.sent_messages), [])

        executor.complete_next()
        self.assertEqual(len(executor.pending), 1)
        self.assertEqual(
            [event.frame_id for event in pipeline_events(enhance.sent_messages)],
            ["A"],
        )
        executor.complete_next()

        observed = pipeline_events(enhance.sent_messages)
        self.assertEqual(
            [(type(event).__name__, getattr(event, "frame_id", None)) for event in observed],
            [("FrameEvent", "A"), ("FrameEvent", "B"), ("EndPassageEvent", None)],
        )
        self.assertEqual(store.get("A"), "enhanced:raw-A")
        self.assertEqual(store.get("B"), "enhanced:raw-B")

        legacy = [
            json.loads(message.content)
            for message in enhance.sent_messages
            if message.ontology == "batch-ready"
        ]
        self.assertEqual(legacy[0]["suitable_count"], 2)
        self.assertEqual(legacy[0]["total_frames"], 2)

        legacy_frames = [
            json.loads(message.content)
            for message in enhance.sent_messages
            if message.ontology == "frame-enhanced"
        ]
        self.assertEqual(
            [(item["frame_id"], item["animal_id"], item["frame_index"]) for item in legacy_frames],
            [("A", "N", 1), ("B", "N", 2)],
        )
        batch_index = next(
            index
            for index, message in enumerate(enhance.sent_messages)
            if message.ontology == "batch-ready"
        )
        self.assertTrue(
            all(
                index < batch_index
                for index, message in enumerate(enhance.sent_messages)
                if message.ontology == "frame-enhanced"
            )
        )

    def test_enhance_error_skips_frame_and_preserves_following_frame_and_end(self):
        store = FrameStore()
        store.put("A", "raw-A")
        store.put("B", "enhance-error")
        store.put("C", "raw-C")
        enhance = make_enhance(store)

        for event in (frame(0, "A"), frame(1, "B"), frame(2, "C"), end(3, total=3)):
            enhance.react(acl_event(event))

        observed = pipeline_events(enhance.sent_messages)
        self.assertEqual(
            [getattr(item, "frame_id", "END") for item in observed],
            ["A", "C", "END"],
        )
        self.assertIsNone(store.get("B"))
        batch_ready = next(
            json.loads(message.content)
            for message in enhance.sent_messages
            if message.ontology == "batch-ready"
        )
        self.assertEqual(batch_ready["suitable_count"], 2)

    def test_consecutive_passages_and_end_pipeline_remain_ordered(self):
        store = FrameStore()
        store.put("N-A", "raw-N")
        store.put("N1-A", "raw-N1")
        enhance = make_enhance(store)

        events = (
            frame(0, "N-A", "N", 1),
            end(1, "N", 1),
            frame(2, "N1-A", "N+1", 1),
            end(3, "N+1", 1),
            EndPipelineEvent(4),
        )
        for event in events:
            enhance.react(acl_event(event))

        observed = pipeline_events(enhance.sent_messages)
        self.assertEqual(
            [(type(item).__name__, getattr(item, "passage_id", None)) for item in observed],
            [
                ("FrameEvent", "N"),
                ("EndPassageEvent", "N"),
                ("FrameEvent", "N+1"),
                ("EndPassageEvent", "N+1"),
                ("EndPipelineEvent", None),
            ],
        )
        self.assertEqual([item.stream_seq for item in observed], list(range(5)))
        self.assertTrue(enhance.inbox.closed)

    def test_logging_failure_releases_once_and_continues_to_end(self):
        store = FrameStore()
        store.put("A", "raw-A")
        store.put("C", "raw-C")
        enhance = make_enhance(store)
        finish_calls = []
        original_finish = enhance._finish_current_frame

        def tracked_finish(event):
            finish_calls.append(event.frame_id)
            original_finish(event)

        enhance._finish_current_frame = tracked_finish
        with patch(
            "mas.agents.data_enhance_agent.display_message",
            side_effect=RuntimeError("synthetic logging error"),
        ):
            for event in (frame(0, "A"), frame(1, "C"), end(2)):
                enhance.react(acl_event(event))

        self.assertEqual(
            [
                getattr(item, "frame_id", "END")
                for item in pipeline_events(enhance.sent_messages)
            ],
            ["A", "C", "END"],
        )
        self.assertEqual(finish_calls, ["A", "C"])
        self.assertFalse(enhance._processing)
        self.assertIsNone(enhance._active_frame_seq)

    def test_errback_cleanup_failure_releases_once_and_continues(self):
        store = FrameStore()
        store.put("B", "enhance-error")
        store.put("C", "raw-C")
        original_discard = store.discard

        def failing_discard(frame_id):
            if frame_id == "B":
                raise RuntimeError("synthetic cleanup error")
            return original_discard(frame_id)

        store.discard = failing_discard
        enhance = make_enhance(store)
        finish_calls = []
        original_finish = enhance._finish_current_frame

        def tracked_finish(event):
            finish_calls.append(event.frame_id)
            original_finish(event)

        enhance._finish_current_frame = tracked_finish
        for event in (frame(0, "B"), frame(1, "C"), end(2)):
            enhance.react(acl_event(event))

        self.assertEqual(
            [
                getattr(item, "frame_id", "END")
                for item in pipeline_events(enhance.sent_messages)
            ],
            ["C", "END"],
        )
        self.assertEqual(finish_calls, ["B", "C"])
        self.assertFalse(enhance._processing)


class OrderedStagesIntegrationTests(unittest.TestCase):
    def test_out_of_order_acl_and_slow_frames_keep_end_after_all_valid_frames(self):
        store = FrameStore()
        store.put("A", "raw-A")
        store.put("B", "raw-B")
        selection_executor = ManualExecutor()
        selection = make_selection(
            store,
            {"raw-A": True, "raw-B": True},
            inbox=OrderedInbox(expected_seq=10),
            executor=selection_executor,
        )

        for event in (frame(10, "A", capture_index=1), end(12), frame(11, "B", capture_index=2)):
            selection.react(acl_event(event))
        selection_executor.complete_next()
        selection_executor.complete_next()

        selection_output = [
            message
            for message in selection.sent_messages
            if message.ontology == "pipeline-event"
        ]
        enhance_executor = ManualExecutor()
        enhance = make_enhance(store, executor=enhance_executor)

        for index in (0, 2, 1):
            selection_output[index].set_sender(selection.aid)
            enhance.react(selection_output[index])
        enhance_executor.complete_next()
        enhance_executor.complete_next()

        observed = pipeline_events(enhance.sent_messages)
        self.assertEqual(
            [getattr(item, "frame_id", "END") for item in observed],
            ["A", "B", "END"],
        )

    def test_selection_and_enhance_match_thread_stage_results(self):
        decisions = {"raw-A": True, "raw-B": False, "raw-C": True}
        selection_adapter = FakeSelectionAdapter(decisions)
        enhance_adapter = FakeEnhanceAdapter()

        q1 = queue.Queue()
        q2 = queue.Queue()
        q3 = queue.Queue()
        for index, frame_id in enumerate(("A", "B", "C"), start=1):
            q1.put({
                "frame_id": frame_id,
                "animal_id": "N",
                "frame_index": index,
                "elapsed_time": float(index * 100),
                "depth_filename": f"{frame_id}.png",
                "label": "suited",
                "img": f"raw-{frame_id}",
            })
        q1.put((_END_ANIMAL, "N", 3, "first-N", "last-N"))
        q1.put(None)

        thread = ThreadPipeline(pid="test", mode="single", fps=1.0)
        thread._select_loop(selection_adapter, q1, q2)
        thread._enhance_loop(enhance_adapter, q2, q3)

        thread_output = []
        while True:
            item = q3.get_nowait()
            if item is None:
                break
            thread_output.append(item)

        store = FrameStore()
        for frame_id in ("A", "B", "C"):
            store.put(frame_id, f"raw-{frame_id}")
        selection = make_selection(store, decisions)
        enhance = make_enhance(store)
        def forward_to_enhance(message):
            message.set_sender(selection.aid)
            enhance.react(message)

        selection.send = forward_to_enhance

        pade_input = (
            frame(0, "A", capture_index=1),
            frame(1, "B", capture_index=2),
            frame(2, "C", capture_index=3),
            end(3, total=3),
        )
        for event in pade_input:
            selection.react(acl_event(event))

        pade_output = pipeline_events(enhance.sent_messages)
        thread_ids = [
            item["frame_id"]
            for item in thread_output
            if isinstance(item, dict)
        ]
        pade_ids = [
            item.frame_id
            for item in pade_output
            if isinstance(item, FrameEvent)
        ]

        self.assertEqual(thread_ids, pade_ids)
        self.assertEqual(thread_ids, ["A", "C"])
        self.assertEqual(
            [store.get(frame_id) for frame_id in pade_ids],
            ["enhanced:raw-A", "enhanced:raw-C"],
        )
        self.assertIsInstance(thread_output[-1], tuple)
        self.assertIsInstance(pade_output[-1], EndPassageEvent)
        self.assertEqual(thread_output[-1][1], pade_output[-1].passage_id)

    def test_each_stage_owns_an_independent_output_sequence(self):
        store = FrameStore()
        store.put("A", "raw-A")
        selection = make_selection(
            store,
            {"raw-A": True},
            inbox=OrderedInbox(expected_seq=10),
            sequencer=StreamSequencer(start=20),
        )
        enhance = make_enhance(
            store,
            inbox=OrderedInbox(expected_seq=20),
            sequencer=StreamSequencer(start=30),
        )
        def forward_to_enhance(message):
            message.set_sender(selection.aid)
            enhance.react(message)

        selection.send = forward_to_enhance

        selection.react(acl_event(frame(10, "A", capture_index=1)))
        selection.react(acl_event(end(11, total=1)))

        observed = pipeline_events(enhance.sent_messages)
        self.assertEqual([item.stream_seq for item in observed], [30, 31])


if __name__ == "__main__":
    unittest.main()
