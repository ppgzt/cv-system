import csv
import queue
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from infra.profiling.telemetry import (
    CAPTURE_TIMING_HEADER,
    HARDWARE_TELEMETRY_HEADER,
    QUEUE_TELEMETRY_HEADER,
    CaptureTimingRecorder,
    HardwareTelemetryMonitor,
    QueueTelemetryMonitor,
    TelemetryContext,
    decode_throttled_mask,
    parse_arm_clock,
    parse_throttled_mask,
    read_arm_clock,
    read_throttled,
)


class CompletedCommand:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


class ThrottlingDecodingTests(unittest.TestCase):
    def test_each_current_and_historical_bit_is_independent(self):
        bits = {
            0: "undervoltage_current",
            1: "arm_frequency_capped_current",
            2: "throttled_current",
            3: "soft_temperature_limit_current",
            16: "undervoltage_occurred",
            17: "arm_frequency_capping_occurred",
            18: "throttling_occurred",
            19: "soft_temperature_limit_occurred",
        }

        for bit, expected_name in bits.items():
            with self.subTest(bit=bit):
                decoded = decode_throttled_mask(1 << bit)
                self.assertTrue(decoded[expected_name])
                self.assertEqual(
                    [name for name, value in decoded.items() if value],
                    [expected_name],
                )

    def test_combined_mask_preserves_current_historical_separation(self):
        mask = (1 << 0) | (1 << 2) | (1 << 17) | (1 << 19)
        decoded = decode_throttled_mask(mask)

        self.assertTrue(decoded["undervoltage_current"])
        self.assertTrue(decoded["throttled_current"])
        self.assertTrue(decoded["arm_frequency_capping_occurred"])
        self.assertTrue(decoded["soft_temperature_limit_occurred"])
        self.assertFalse(decoded["undervoltage_occurred"])
        self.assertFalse(decoded["throttling_occurred"])


class VcgencmdParsingTests(unittest.TestCase):
    def test_parse_arm_clock(self):
        self.assertEqual(parse_arm_clock("frequency(48)=2400000000"), 2_400_000_000)
        self.assertIsNone(parse_arm_clock(""))
        self.assertIsNone(parse_arm_clock(None))
        self.assertIsNone(parse_arm_clock("frequency=2400000000"))
        self.assertIsNone(parse_arm_clock("frequency(48)=not-a-number"))

    def test_parse_throttled_mask(self):
        self.assertEqual(parse_throttled_mask("throttled=0x0"), 0)
        self.assertEqual(parse_throttled_mask("throttled=0x50005"), 0x50005)
        self.assertIsNone(parse_throttled_mask(""))
        self.assertIsNone(parse_throttled_mask("invalid=0x1"))

    @patch("infra.profiling.telemetry.subprocess.run")
    def test_read_arm_clock_valid_command(self, run):
        run.return_value = CompletedCommand("frequency(48)=2400000000\n")

        reading = read_arm_clock(timeout=0.25)

        self.assertEqual(reading["arm_clock_hz"], 2_400_000_000)
        self.assertTrue(reading["clock_command_available"])
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["vcgencmd", "measure_clock", "arm"])
        self.assertEqual(kwargs["timeout"], 0.25)
        self.assertNotIn("shell", kwargs)

    def test_read_arm_clock_failures_are_unavailable_not_zero(self):
        failures = [
            CompletedCommand(""),
            CompletedCommand("unexpected"),
            CompletedCommand("frequency(48)=2400000000", returncode=1),
            FileNotFoundError(),
            subprocess.TimeoutExpired(["vcgencmd"], 0.1),
        ]
        for outcome in failures:
            with self.subTest(outcome=type(outcome).__name__):
                with patch("infra.profiling.telemetry.subprocess.run") as run:
                    if isinstance(outcome, BaseException):
                        run.side_effect = outcome
                    else:
                        run.return_value = outcome
                    reading = read_arm_clock(timeout=0.1)
                self.assertIsNone(reading["arm_clock_hz"])
                self.assertFalse(reading["clock_command_available"])

    @patch("infra.profiling.telemetry.subprocess.run")
    def test_read_throttled_zero_and_nonzero(self, run):
        run.return_value = CompletedCommand("throttled=0x0\n")
        zero = read_throttled()
        self.assertEqual(zero["throttled_raw"], "throttled=0x0")
        self.assertEqual(zero["throttled_mask"], 0)
        self.assertTrue(zero["throttling_command_available"])
        self.assertFalse(any(
            value for key, value in zero.items()
            if key.endswith("_current") or key.endswith("_occurred")
        ))

        run.return_value = CompletedCommand("throttled=0x50005")
        nonzero = read_throttled()
        self.assertEqual(nonzero["throttled_raw"], "throttled=0x50005")
        self.assertEqual(nonzero["throttled_mask"], 0x50005)
        self.assertTrue(nonzero["throttled_current"])
        self.assertTrue(nonzero["undervoltage_current"])
        self.assertTrue(nonzero["undervoltage_occurred"])
        self.assertTrue(nonzero["throttling_occurred"])

    def test_read_throttled_failures_preserve_unknown_state(self):
        failures = [
            CompletedCommand("not-throttled=0x1"),
            CompletedCommand("throttled=0x1", returncode=1),
            FileNotFoundError(),
            subprocess.TimeoutExpired(["vcgencmd"], 0.1),
        ]
        for outcome in failures:
            with self.subTest(outcome=type(outcome).__name__):
                with patch("infra.profiling.telemetry.subprocess.run") as run:
                    if isinstance(outcome, BaseException):
                        run.side_effect = outcome
                    else:
                        run.return_value = outcome
                    reading = read_throttled(timeout=0.1)
                self.assertFalse(reading["throttling_command_available"])
                self.assertIsNone(reading["throttled_mask"])
                self.assertIsNone(reading["throttled_current"])
                self.assertIsNone(reading["throttling_occurred"])


class TelemetryMonitorTests(unittest.TestCase):
    def test_capture_timing_two_phase_admission_uses_receiver_timestamp(self):
        context = TelemetryContext(
            "capture-run", "pade_fixed_fps", 2.0,
            monotonic_origin_ns=1_000_000_000,
        )
        recorder = CaptureTimingRecorder(context)

        self.assertTrue(recorder.register_scheduled_event(
            passage_id="N",
            capture_index=1,
            frame_id="frame",
            source_filename="source.png",
            source_relative_time_ms=500.0,
            scheduled_capture_time_ms=500.0,
            scheduled_monotonic_ns=1_500_000_000,
        ))
        self.assertEqual(recorder.pending_count(), 1)
        self.assertTrue(recorder.record_admission("frame", 1_513_000_000))

        self.assertEqual(recorder.pending_count(), 0)
        rows = recorder.get_all_data()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["actual_enqueue_monotonic_ns"], 1_513_000_000)
        self.assertEqual(rows[0]["lateness_ms"], 13.0)

    def test_capture_timing_csv_header_and_nullable_values(self):
        with tempfile.TemporaryDirectory() as reports_dir:
            context = TelemetryContext(
                "capture-run", "original_timing", None, monotonic_origin_ns=100
            )
            recorder = CaptureTimingRecorder(context, reports_dir=reports_dir)
            self.assertTrue(recorder.record(
                passage_id="N",
                capture_index=1,
                frame_id="frame",
                source_filename="source.png",
                source_relative_time_ms=20.0,
                scheduled_capture_time_ms=20.0,
                scheduled_monotonic_ns=200,
                actual_enqueue_monotonic_ns=250,
            ))
            self.assertTrue(recorder.persist())

            csv_path = Path(reports_dir) / "capture-run" / "capture_timing.csv"
            with csv_path.open(newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, CAPTURE_TIMING_HEADER)
                row = next(reader)
            self.assertEqual(row["capture_fps"], "")
            self.assertEqual(row["source_filename"], "source.png")
            self.assertEqual(row["actual_enqueue_monotonic_ns"], "250")

    def test_queue_monitor_records_without_consuming_or_reordering(self):
        q1 = queue.Queue()
        q2 = queue.Queue()
        q3 = queue.Queue()
        original_q1 = ["frame", ("END_ANIMAL", "N"), None]
        for item in original_q1:
            q1.put(item)
        q2.put("enhance-frame")

        with tempfile.TemporaryDirectory() as reports_dir:
            origin = time.monotonic_ns()
            context = TelemetryContext(
                run_id="queue-run",
                condition="opaque-condition",
                capture_fps=5.0,
                monotonic_origin_ns=origin,
            )
            context.set_capture_passage_id("N")
            monitor = QueueTelemetryMonitor(
                context, q1, q2, q3, interval=0.01, reports_dir=reports_dir
            )
            monitor.start()
            time.sleep(0.035)
            monitor.stop()
            monitor.join(timeout=1.0)

            self.assertFalse(monitor.is_alive())
            rows = monitor.get_all_data()
            self.assertGreaterEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["run_id"], "queue-run")
                self.assertEqual(row["condition"], "opaque-condition")
                self.assertEqual(row["capture_fps"], 5.0)
                self.assertEqual(row["capture_passage_id"], "N")
                self.assertEqual(row["capture_to_selection_qsize"], 3)
                self.assertEqual(row["selection_to_preprocessing_qsize"], 1)
                self.assertEqual(row["preprocessing_to_prediction_qsize"], 0)
                self.assertIsInstance(row["monotonic_ns"], int)
                self.assertGreaterEqual(row["elapsed_s"], 0.0)

            csv_path = Path(reports_dir) / "queue-run" / "queue_telemetry.csv"
            with open(csv_path, newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, QUEUE_TELEMETRY_HEADER)
                self.assertGreaterEqual(len(list(reader)), 2)

        self.assertEqual([q1.get_nowait() for _ in original_q1], original_q1)
        self.assertEqual(q2.get_nowait(), "enhance-frame")
        self.assertTrue(q3.empty())

    def test_hardware_monitor_writes_raw_flags_booleans_and_nulls(self):
        def clock_reader(_timeout):
            return {"arm_clock_hz": None, "clock_command_available": False}

        def throttling_reader(_timeout):
            return {
                "throttled_raw": "throttled=0x50005",
                "throttled_mask": 0x50005,
                **decode_throttled_mask(0x50005),
                "throttling_command_available": True,
            }

        with tempfile.TemporaryDirectory() as reports_dir:
            context = TelemetryContext(
                run_id="hardware-run",
                condition="visual_adaptive",
                capture_fps=None,
            )
            monitor = HardwareTelemetryMonitor(
                context,
                interval=0.01,
                reports_dir=reports_dir,
                clock_reader=clock_reader,
                throttling_reader=throttling_reader,
            )
            monitor.start()
            time.sleep(0.035)
            monitor.stop()
            monitor.join(timeout=1.0)

            self.assertFalse(monitor.is_alive())
            csv_path = (
                Path(reports_dir) / "hardware-run" / "hardware_telemetry.csv"
            )
            with open(csv_path, newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, HARDWARE_TELEMETRY_HEADER)
                rows = list(reader)
            self.assertGreaterEqual(len(rows), 2)
            self.assertEqual(rows[0]["condition"], "visual_adaptive")
            self.assertEqual(rows[0]["capture_fps"], "")
            self.assertEqual(rows[0]["arm_clock_hz"], "")
            self.assertEqual(rows[0]["clock_command_available"], "False")
            self.assertEqual(rows[0]["throttled_raw"], "throttled=0x50005")
            self.assertEqual(rows[0]["throttled_current"], "True")
            self.assertEqual(rows[0]["arm_frequency_capped_current"], "False")
            self.assertEqual(rows[0]["throttling_command_available"], "True")

    def test_hardware_reader_exceptions_do_not_kill_monitor(self):
        def failing_reader(_timeout):
            raise RuntimeError("unavailable")

        with tempfile.TemporaryDirectory() as reports_dir:
            context = TelemetryContext("run", "condition", None)
            monitor = HardwareTelemetryMonitor(
                context,
                interval=0.01,
                reports_dir=reports_dir,
                clock_reader=failing_reader,
                throttling_reader=failing_reader,
            )
            monitor.start()
            time.sleep(0.025)
            monitor.stop()
            monitor.join(timeout=1.0)

            rows = monitor.get_all_data()
            self.assertGreaterEqual(len(rows), 1)
            self.assertIsNone(rows[0]["arm_clock_hz"])
            self.assertFalse(rows[0]["clock_command_available"])
            self.assertIsNone(rows[0]["throttled_current"])
            self.assertFalse(rows[0]["throttling_command_available"])


if __name__ == "__main__":
    unittest.main()
