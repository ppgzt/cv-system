import importlib.util
import pathlib
import unittest


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "data-analysis"
    / "visual_activity_calibration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "visual_activity_calibration",
    MODULE_PATH,
)
CALIBRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATION)


class VisualActivityCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "passage_id": "N",
                "capture_index": 1,
                "timestamp_ms": 0.0,
                "delta_t_ms": None,
                "depth_filename": "n0.png",
                "label": "background",
                "mad": None,
            },
            {
                "passage_id": "N",
                "capture_index": 2,
                "timestamp_ms": 100.0,
                "delta_t_ms": 100.0,
                "depth_filename": "n1.png",
                "label": "background",
                "mad": 20.0,
            },
            {
                "passage_id": "N",
                "capture_index": 3,
                "timestamp_ms": 200.0,
                "delta_t_ms": 100.0,
                "depth_filename": "n2.png",
                "label": "suited",
                "mad": 0.0,
            },
            {
                "passage_id": "N",
                "capture_index": 4,
                "timestamp_ms": 300.0,
                "delta_t_ms": 100.0,
                "depth_filename": "n3.png",
                "label": "suited",
                "mad": 0.0,
            },
            {
                "passage_id": "M",
                "capture_index": 1,
                "timestamp_ms": 0.0,
                "delta_t_ms": None,
                "depth_filename": "m0.png",
                "label": "suited",
                "mad": None,
            },
        ]

    def test_passage_coverage_retention_active_ratio_and_delay_are_distinct(self):
        summary, rows = CALIBRATION.evaluate_configuration(
            self.records,
            threshold=10.0,
            patience=2,
        )

        self.assertEqual(summary["n_suited_passages"], 2)
        self.assertEqual(summary["n_suited_passages_covered"], 1)
        self.assertEqual(summary["suited_passage_coverage"], 0.5)
        self.assertAlmostEqual(summary["suited_frame_retention"], 1 / 3)
        self.assertAlmostEqual(summary["active_ratio"], 2 / 5)
        self.assertEqual(summary["activation_delay_median_ms"], -100.0)
        self.assertEqual(summary["n_missed_passages"], 1)
        self.assertEqual(summary["missed_passage_ids"], "M")
        self.assertEqual(len(rows), 2)

    def test_distribution_uses_human_labels_and_handles_empty_groups(self):
        rows = CALIBRATION.mad_distribution(self.records)
        by_group = {row["group"]: row for row in rows}

        self.assertEqual(by_group["global"]["n_pairs"], 3)
        self.assertEqual(by_group["suited"]["n_pairs"], 2)
        self.assertEqual(by_group["background"]["n_pairs"], 1)
        self.assertEqual(by_group["ruido"]["n_pairs"], 0)
        self.assertIsNone(by_group["ruido"]["median"])

    def test_always_active_is_explicit_baseline(self):
        baseline = CALIBRATION.always_active_baseline(self.records)
        self.assertEqual(baseline["configuration"], "always_active")
        self.assertEqual(baseline["suited_passage_coverage"], 1.0)
        self.assertEqual(baseline["suited_frame_retention"], 1.0)
        self.assertEqual(baseline["active_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
