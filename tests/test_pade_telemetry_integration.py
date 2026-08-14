import csv
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

import mas  # noqa: F401  (expoe PADE vendorizado no sys.path)

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage

from domain.pipeline_events import (
    EndPassageEvent,
    EndPipelineEvent,
    FrameEvent,
    event_to_json,
)
from infra.profiling.telemetry import (
    HARDWARE_TELEMETRY_HEADER,
    QUEUE_TELEMETRY_HEADER,
    CaptureTimingRecorder,
    TelemetryContext,
)
from mas.agents.frame_selection import FrameSelectionAgent
from mas.infrastructure.ordered_inbox import OrderedInbox
from mas.infrastructure.pade_telemetry import PadeTelemetrySession


def frame(seq, frame_id="frame", passage_id="N"):
    return FrameEvent(
        stream_seq=seq,
        frame_id=frame_id,
        passage_id=passage_id,
        capture_index=1,
        elapsed_time=500.0,
        depth_filename="source.png",
        label="suited",
        dataset_timestamp_ms=None,
    )


def end(seq, passage_id="N"):
    return EndPassageEvent(
        stream_seq=seq,
        passage_id=passage_id,
        total_captured_frames=1,
        first_capture_time="first",
        last_capture_time="last",
    )


def acl_event(event):
    message = ACLMessage(ACLMessage.INFORM)
    message.set_ontology("pipeline-event")
    message.set_sender(AID(name="capture@localhost:5003"))
    message.set_content(event_to_json(event))
    return message


class PadeTelemetryIntegrationTests(unittest.TestCase):
    def test_direct_inbox_mapping_counts_ready_and_reorder_without_consuming(self):
        selection = OrderedInbox(expected_seq=10)
        enhance = OrderedInbox()
        prediction = OrderedInbox()

        selection.put(frame(10, "processing"))
        self.assertEqual(selection.get(block=False).frame_id, "processing")
        selection.put(frame(12, "future"))
        enhance.put(frame(0, "enhance"))

        with tempfile.TemporaryDirectory() as reports_dir:
            session = PadeTelemetrySession(
                run_id="mapping",
                condition="pade_fixed_fps",
                capture_fps=5.0,
                monotonic_origin_ns=time.monotonic_ns(),
                selection_inbox=selection,
                enhance_inbox=enhance,
                prediction_inbox=prediction,
                reports_dir=reports_dir,
                queue_interval=0.01,
                hardware_interval=0.01,
                clock_reader=lambda _timeout: {
                    "arm_clock_hz": None,
                    "clock_command_available": False,
                },
                throttling_reader=lambda _timeout: {
                    "throttled_raw": None,
                    "throttled_mask": None,
                    "undervoltage_current": None,
                    "arm_frequency_capped_current": None,
                    "throttled_current": None,
                    "soft_temperature_limit_current": None,
                    "undervoltage_occurred": None,
                    "arm_frequency_capping_occurred": None,
                    "throttling_occurred": None,
                    "soft_temperature_limit_occurred": None,
                    "throttling_command_available": False,
                },
            )
            session.start()
            time.sleep(0.025)
            session.stop()

            row = session.queue_monitor.get_all_data()[0]
            self.assertEqual(row["capture_to_selection_qsize"], 1)
            self.assertEqual(row["selection_to_preprocessing_qsize"], 1)
            self.assertEqual(row["preprocessing_to_prediction_qsize"], 0)
            self.assertEqual(selection.ready_qsize(), 0)
            self.assertEqual(selection.reorder_buffer_size(), 1)

            selection.put(frame(11, "gap"))
            self.assertEqual(selection.qsize(), 2)
            self.assertEqual(selection.ready_qsize(), 2)
            self.assertEqual(selection.reorder_buffer_size(), 0)

            output = Path(reports_dir) / "mapping"
            with (output / "queue_telemetry.csv").open(newline="") as file:
                self.assertEqual(csv.DictReader(file).fieldnames, QUEUE_TELEMETRY_HEADER)
            with (output / "hardware_telemetry.csv").open(newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, HARDWARE_TELEMETRY_HEADER)
                rows = list(reader)
                self.assertEqual(rows[0]["clock_command_available"], "False")
                self.assertEqual(rows[0]["throttled_current"], "")
            self.assertFalse(session.queue_monitor.is_alive())
            self.assertFalse(session.hardware_monitor.is_alive())

    def test_admission_is_recorded_after_selection_inbox_put(self):
        context = TelemetryContext(
            "admission", "pade_fixed_fps", 2.0,
            monotonic_origin_ns=1_000_000_000,
        )
        recorder = CaptureTimingRecorder(context)
        recorder.register_scheduled_event(
            passage_id="N",
            capture_index=1,
            frame_id="frame",
            source_filename="source.png",
            source_relative_time_ms=500.0,
            scheduled_capture_time_ms=500.0,
            scheduled_monotonic_ns=1_500_000_000,
        )
        selection = FrameSelectionAgent(
            aid=AID(name="selection@localhost:5005"),
            frame_selection_adapter=object(),
            next_agent_aid="enhance@localhost:5004",
            telemetry_context=context,
            capture_timing_recorder=recorder,
            monotonic_ns=lambda: 1_513_000_000,
        )
        selection.send = lambda _message: None
        # Mantém o evento admitido sem iniciar adapter; o teste isola o ponto
        # receptor inbox.put -> timestamp.
        selection._processing = True
        selection.react(acl_event(frame(0)))

        self.assertEqual(selection.inbox.qsize(), 1)
        rows = recorder.get_all_data()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["lateness_ms"], 13.0)

    def test_passage_context_clear_is_conditional_and_never_waits_downstream(self):
        context = TelemetryContext("run", "pade_original_timing", None)
        selection = FrameSelectionAgent(
            aid=AID(name="selection@localhost:5005"),
            frame_selection_adapter=object(),
            next_agent_aid="enhance@localhost:5004",
            telemetry_context=context,
        )
        selection.send = lambda _message: None
        selection._processing = True

        context.set_capture_passage_id("N")
        selection.react(acl_event(end(0, "N")))
        self.assertIsNone(
            context.sample_metadata(time.monotonic_ns())["capture_passage_id"]
        )

        context.set_capture_passage_id("N+1")
        # END N atrasado não pode apagar a admissão já ativa de N+1.
        late_selection = FrameSelectionAgent(
            aid=AID(name="selection2@localhost:5015"),
            frame_selection_adapter=object(),
            next_agent_aid="enhance@localhost:5004",
            telemetry_context=context,
        )
        late_selection.send = lambda _message: None
        late_selection._processing = True
        late_selection.react(acl_event(end(0, "N")))
        self.assertEqual(
            context.sample_metadata(time.monotonic_ns())["capture_passage_id"],
            "N+1",
        )

    def test_monitoring_does_not_change_logical_inbox_output(self):
        events = [frame(0, "A"), end(1)]

        def consume(inbox):
            return [inbox.get(block=False) for _ in events]

        telemetry_off = OrderedInbox()
        for event in events:
            telemetry_off.put(event)
        off_output = consume(telemetry_off)

        telemetry_on = OrderedInbox()
        unused_2 = OrderedInbox()
        unused_3 = OrderedInbox()
        for event in events:
            telemetry_on.put(event)
        with tempfile.TemporaryDirectory() as reports_dir:
            session = PadeTelemetrySession(
                run_id="non-interference",
                condition="pade_original_timing",
                capture_fps=None,
                monotonic_origin_ns=time.monotonic_ns(),
                selection_inbox=telemetry_on,
                enhance_inbox=unused_2,
                prediction_inbox=unused_3,
                reports_dir=reports_dir,
                capture_timing_enabled=False,
                queue_interval=0.01,
                hardware_interval=1.0,
            )
            session.start()
            on_output = consume(telemetry_on)
            session.stop()

        self.assertEqual(on_output, off_output)

    def test_monitors_cover_downstream_drain_until_global_shutdown(self):
        selection = OrderedInbox()
        enhance = OrderedInbox()
        prediction = OrderedInbox()
        with tempfile.TemporaryDirectory() as reports_dir:
            session = PadeTelemetrySession(
                run_id="drain",
                condition="pade_fixed_fps",
                capture_fps=10.0,
                monotonic_origin_ns=time.monotonic_ns(),
                selection_inbox=selection,
                enhance_inbox=enhance,
                prediction_inbox=prediction,
                reports_dir=reports_dir,
                queue_interval=0.005,
                hardware_interval=0.005,
                clock_reader=lambda _timeout: {
                    "arm_clock_hz": None,
                    "clock_command_available": False,
                },
                throttling_reader=lambda _timeout: {
                    "throttled_raw": None,
                    "throttled_mask": None,
                    "undervoltage_current": None,
                    "arm_frequency_capped_current": None,
                    "throttled_current": None,
                    "soft_temperature_limit_current": None,
                    "undervoltage_occurred": None,
                    "arm_frequency_capping_occurred": None,
                    "throttling_occurred": None,
                    "soft_temperature_limit_occurred": None,
                    "throttling_command_available": False,
                },
            )
            session.start()
            selection.put(frame(0, "A"))
            selection.put(end(1))
            selection.put(EndPipelineEvent(stream_seq=2))
            time.sleep(0.012)

            # Captura terminou, mas o trabalho agora está no downstream.
            while selection.qsize():
                event = selection.get(block=False)
                enhance.put(replace(event, stream_seq=enhance.expected_seq))
            while enhance.qsize():
                event = enhance.get(block=False)
                prediction.put(
                    replace(event, stream_seq=prediction.expected_seq)
                )
            time.sleep(0.012)

            self.assertTrue(any(
                row["preprocessing_to_prediction_qsize"] > 0
                for row in session.queue_monitor.get_all_data()
            ))
            while prediction.qsize():
                prediction.get(block=False)
            session.stop()

            output = Path(reports_dir) / "drain"
            self.assertTrue((output / "queue_telemetry.csv").exists())
            self.assertTrue((output / "hardware_telemetry.csv").exists())
            self.assertTrue((output / "capture_timing.csv").exists())


if __name__ == "__main__":
    unittest.main()
