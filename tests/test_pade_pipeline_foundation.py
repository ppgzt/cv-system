import queue
import threading
import unittest

import numpy as np

from domain.pipeline_events import (
    EndPassageEvent,
    EndPipelineEvent,
    FrameEvent,
    event_from_dict,
    event_from_json,
    event_to_dict,
    event_to_json,
)
from mas.infrastructure.frame_store import FrameStore
from mas.infrastructure.ordered_inbox import OrderedInbox, OrderedInboxClosed
from mas.infrastructure.stream_sequence import StreamSequencer


def frame(seq: int, passage_id: str = "N") -> FrameEvent:
    return FrameEvent(
        stream_seq=seq,
        frame_id=f"frame-{seq}",
        passage_id=passage_id,
        capture_index=seq,
        elapsed_time=float(seq * 100),
        depth_filename=f"depth-{seq}.png",
        label="suited",
        dataset_timestamp_ms=float(seq * 100),
    )


class PipelineEventTests(unittest.TestCase):
    def test_round_trip_preserves_each_event_type_and_metadata(self):
        events = [
            frame(7),
            EndPassageEvent(
                stream_seq=8,
                passage_id="N",
                total_captured_frames=7,
                first_capture_time="2026-01-01T00:00:00",
                last_capture_time="2026-01-01T00:00:05",
            ),
            EndPipelineEvent(stream_seq=9),
        ]

        for event in events:
            with self.subTest(event=type(event).__name__):
                self.assertEqual(event_from_dict(event_to_dict(event)), event)
                self.assertEqual(event_from_json(event_to_json(event)), event)

    def test_frame_event_payload_does_not_contain_image_data(self):
        payload = event_to_dict(frame(1))

        self.assertEqual(payload["event_type"], "frame")
        self.assertEqual(payload["passage_id"], "N")
        self.assertEqual(payload["stream_seq"], 1)
        self.assertNotIn("img", payload)
        self.assertNotIn("image", payload)

    def test_unknown_event_type_is_rejected(self):
        with self.assertRaises(ValueError):
            event_from_dict({"event_type": "unknown", "stream_seq": 1})


class StreamSequencerTests(unittest.TestCase):
    def test_each_logical_edge_has_an_independent_sequence(self):
        capture_to_selection = StreamSequencer(start=10)
        selection_to_preprocessing = StreamSequencer(start=10)

        self.assertEqual(capture_to_selection.next_seq(), 10)
        self.assertEqual(capture_to_selection.next_seq(), 11)
        self.assertEqual(selection_to_preprocessing.next_seq(), 10)

    def test_concurrent_allocations_are_unique_and_contiguous(self):
        sequencer = StreamSequencer()
        allocated = []
        allocated_lock = threading.Lock()

        def allocate():
            local = [sequencer.next_seq() for _ in range(100)]
            with allocated_lock:
                allocated.extend(local)

        threads = [threading.Thread(target=allocate) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(sorted(allocated), list(range(800)))
        self.assertEqual(sequencer.next_value, 800)


class OrderedInboxTests(unittest.TestCase):
    def test_in_order_arrival_preserves_order(self):
        inbox = OrderedInbox(expected_seq=1)
        for seq in (1, 2, 3, 4):
            inbox.put(frame(seq))

        self.assertEqual(
            [inbox.get(block=False).stream_seq for _ in range(4)],
            [1, 2, 3, 4],
        )

    def test_out_of_order_arrival_is_released_in_stream_order(self):
        inbox = OrderedInbox(expected_seq=1)
        for seq in (1, 3, 2, 4):
            inbox.put(frame(seq))

        self.assertEqual(
            [inbox.get(block=False).stream_seq for _ in range(4)],
            [1, 2, 3, 4],
        )

    def test_end_passage_cannot_overtake_a_missing_frame(self):
        inbox = OrderedInbox(expected_seq=10)
        first = frame(10)
        end = EndPassageEvent(12, "N", 2, "first", "last")
        second = frame(11)

        inbox.put(first)
        inbox.put(end)
        self.assertEqual(inbox.qsize(), 2)
        self.assertEqual(inbox.ready_qsize(), 1)
        self.assertEqual(inbox.reorder_buffer_size(), 1)
        inbox.put(second)

        observed = [inbox.get(block=False) for _ in range(3)]
        self.assertEqual(observed, [first, second, end])

    def test_consecutive_passages_remain_in_logical_order(self):
        inbox = OrderedInbox(expected_seq=20)
        events = [
            frame(20, "N"),
            EndPassageEvent(21, "N", 1, "first-n", "last-n"),
            frame(22, "N+1"),
        ]
        for event in events:
            inbox.put(event)

        self.assertEqual(
            [inbox.get(block=False) for _ in range(len(events))],
            events,
        )

    def test_end_pipeline_waits_for_every_previous_sequence(self):
        inbox = OrderedInbox(expected_seq=30)
        end = EndPipelineEvent(32)
        first = frame(30)
        second = frame(31)

        inbox.put(end)
        self.assertEqual(inbox.qsize(), 1)
        self.assertEqual(inbox.ready_qsize(), 0)
        inbox.put(first)
        self.assertEqual(inbox.qsize(), 2)
        self.assertEqual(inbox.ready_qsize(), 1)
        inbox.put(second)

        self.assertEqual(
            [inbox.get(block=False) for _ in range(3)],
            [first, second, end],
        )

    def test_qsize_counts_total_pending_occupancy(self):
        inbox = OrderedInbox(expected_seq=1)

        inbox.put(frame(3))
        self.assertEqual(inbox.qsize(), 1)
        self.assertEqual(inbox.ready_qsize(), 0)
        self.assertEqual(inbox.reorder_buffer_size(), 1)
        self.assertEqual(inbox.buffered_size(), 1)

        inbox.put(frame(1))
        self.assertEqual(inbox.qsize(), 2)
        self.assertEqual(inbox.ready_qsize(), 1)
        self.assertEqual(inbox.reorder_buffer_size(), 1)
        self.assertEqual(inbox.buffered_size(), inbox.qsize())

        inbox.put(frame(2))
        self.assertEqual(inbox.qsize(), 3)
        self.assertEqual(inbox.ready_qsize(), 3)
        self.assertEqual(inbox.reorder_buffer_size(), 0)
        self.assertEqual(inbox.buffered_size(), 3)

    def test_duplicate_or_already_released_sequence_is_rejected(self):
        inbox = OrderedInbox(expected_seq=1)
        inbox.put(frame(2))
        with self.assertRaises(ValueError):
            inbox.put(frame(2))

        inbox.put(frame(1))
        with self.assertRaises(ValueError):
            inbox.put(frame(1))

    def test_nonblocking_empty_get_uses_queue_empty(self):
        inbox = OrderedInbox()
        with self.assertRaises(queue.Empty):
            inbox.get(block=False)

    def test_close_wakes_blocked_consumer_without_deadlock(self):
        inbox = OrderedInbox()
        outcome = []

        def consume():
            try:
                inbox.get()
            except OrderedInboxClosed:
                outcome.append("closed")

        consumer = threading.Thread(target=consume)
        consumer.start()
        inbox.close()
        consumer.join(timeout=1.0)

        self.assertFalse(consumer.is_alive())
        self.assertEqual(outcome, ["closed"])
        with self.assertRaises(OrderedInboxClosed):
            inbox.put(frame(0))

    def test_close_allows_already_ready_events_to_be_drained(self):
        inbox = OrderedInbox()
        event = frame(0)
        inbox.put(event)
        inbox.close()

        self.assertEqual(inbox.get(), event)
        with self.assertRaises(OrderedInboxClosed):
            inbox.get()


class FrameStoreTests(unittest.TestCase):
    def test_put_get_overwrite_pop_and_discard(self):
        store = FrameStore()
        raw = object()
        enhanced = object()

        store.put("accepted", raw)
        self.assertIs(store.get("accepted"), raw)

        store.put("accepted", enhanced)
        self.assertIs(store.get("accepted"), enhanced)
        self.assertIs(store.pop("accepted"), enhanced)
        self.assertIsNone(store.get("accepted"))

        store.put("rejected", raw)
        self.assertTrue(store.discard("rejected"))
        self.assertFalse(store.discard("rejected"))

    def test_clear_releases_all_frames(self):
        store = FrameStore()
        for index in range(5):
            store.put(str(index), object())

        self.assertEqual(store.clear(), 5)
        self.assertEqual(len(store), 0)

    def test_basic_concurrent_access_uses_one_shared_lock(self):
        store = FrameStore()
        thread_count = 8
        frames_per_thread = 100

        def produce(thread_index: int):
            for frame_index in range(frames_per_thread):
                frame_id = f"{thread_index}-{frame_index}"
                store.put(frame_id, (thread_index, frame_index))

        threads = [
            threading.Thread(target=produce, args=(thread_index,))
            for thread_index in range(thread_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(store), thread_count * frames_per_thread)
        for thread_index in range(thread_count):
            for frame_index in range(frames_per_thread):
                self.assertEqual(
                    store.get(f"{thread_index}-{frame_index}"),
                    (thread_index, frame_index),
                )

    def test_read_lease_is_zero_copy_and_survives_main_entry_changes(self):
        store = FrameStore()
        raw = np.arange(12, dtype=np.uint16).reshape(3, 4)
        enhanced = np.ones((3, 4, 3), dtype=np.float32)
        store.put("frame", raw)
        lease_id = store.retain("frame", owner="visual", passage_id="N")

        leased = store.read_lease(lease_id, owner="visual")
        self.assertIs(leased, raw)
        self.assertTrue(np.shares_memory(raw, leased))

        store.put("frame", enhanced)
        self.assertIs(store.get("frame"), enhanced)
        self.assertIs(store.read_lease(lease_id, owner="visual"), raw)

        self.assertIs(store.pop("frame"), enhanced)
        self.assertIs(store.read_lease(lease_id, owner="visual"), raw)
        self.assertTrue(store.release_lease(lease_id, owner="visual"))
        self.assertFalse(store.release_lease(lease_id, owner="visual"))
        with self.assertRaises(KeyError):
            store.read_lease(lease_id, owner="visual")

    def test_discard_does_not_invalidate_lease_and_owner_is_enforced(self):
        store = FrameStore()
        raw = object()
        store.put("frame", raw)
        lease_id = store.retain("frame", owner="visual", passage_id="N")

        self.assertTrue(store.discard("frame"))
        self.assertIs(store.read_lease(lease_id, owner="visual"), raw)
        with self.assertRaises(PermissionError):
            store.read_lease(lease_id, owner="selection")
        with self.assertRaises(PermissionError):
            store.release_lease(lease_id, owner="selection")

    def test_passage_cleanup_only_releases_matching_visual_leases(self):
        store = FrameStore()
        for frame_id in ("n-a", "n-b", "next", "other"):
            store.put(frame_id, object())
        n_a = store.retain("n-a", owner="visual", passage_id="N")
        n_b = store.retain("n-b", owner="visual", passage_id="N")
        next_id = store.retain("next", owner="visual", passage_id="N+1")
        other = store.retain("other", owner="debug", passage_id="N")

        self.assertEqual(
            store.release_leases(owner="visual", passage_id="N"),
            2,
        )
        self.assertEqual(store.lease_count(owner="visual"), 1)
        self.assertIsNotNone(store.read_lease(next_id, owner="visual"))
        self.assertIsNotNone(store.read_lease(other, owner="debug"))
        for lease_id in (n_a, n_b):
            with self.assertRaises(KeyError):
                store.read_lease(lease_id, owner="visual")


if __name__ == "__main__":
    unittest.main()
