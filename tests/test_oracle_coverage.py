import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "data-analysis" / "oracle_coverage.py"
SPEC = importlib.util.spec_from_file_location("oracle_coverage", SCRIPT_PATH)
oracle_coverage = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = oracle_coverage
SPEC.loader.exec_module(oracle_coverage)


class OracleCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "dataset"
        for tag, frames in {
            "A": [
                (0, "A0.png", "background"),
                (400, "A1.png", "suited"),
                (1000, "A2.png", "ruido"),
            ],
            "B": [
                (0, "B0.png", "background"),
                (1000, "B1.png", "parcial"),
            ],
        }.items():
            index_dir = self.root / "animal-tags" / tag
            depth_dir = self.root / "DEPTH" / tag
            index_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            index = []
            for timestamp, filename, label in frames:
                index.append({
                    "relative_time_ms": timestamp,
                    "depth_filename": filename,
                    "label": label,
                })
                (depth_dir / filename).write_bytes(b"depth")
            (index_dir / "simulation_index.json").write_text(json.dumps(index))

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _selector_template(
        self,
        *,
        mismatched_total=False,
        a_index="2",
        b_index="1",
        uuid_keys=False,
    ):
        path = Path(self.temporary_directory.name) / "selector_2.json"
        path.write_text(json.dumps({
            "animals": {
                "A": {
                    "total_of_images": 4 if mismatched_total else 3,
                    "suitable_images": int(a_index is not None),
                    "imgs": ({
                        "frame-a" if uuid_keys else a_index: {}
                    } if a_index is not None else {}),
                },
                "B": {
                    "total_of_images": 3,
                    "suitable_images": int(b_index is not None),
                    "imgs": ({
                        "frame-b" if uuid_keys else b_index: {}
                    } if b_index is not None else {}),
                },
            }
        }))
        return str(Path(self.temporary_directory.name) / "selector_{fps:g}.json")

    def test_dataset_audit_maps_every_frame_without_heuristics(self):
        indexes, audit = oracle_coverage.audit_and_load_dataset(
            self.root, ["A", "B"]
        )

        self.assertEqual(sorted(indexes), ["A", "B"])
        self.assertEqual(audit.n_dataset_passages, 2)
        self.assertEqual(audit.n_dataset_frames, 5)
        self.assertEqual(audit.n_evaluated_suited_frames, 1)
        self.assertEqual(audit.label_counts["suited"], 1)

    def test_gt_events_unique_frames_and_selector_outcomes_are_distinct(self):
        indexes, _ = oracle_coverage.audit_and_load_dataset(self.root, ["A", "B"])

        summary, rows = oracle_coverage.analyze_oracle_coverage(
            indexes, [2], self._selector_template()
        )
        by_tag = {row["passage_id"]: row for row in rows}

        self.assertEqual(by_tag["A"]["n_capture_events"], 3)
        self.assertEqual(by_tag["A"]["n_human_suited_capture_events"], 1)
        self.assertTrue(by_tag["A"]["gt_suited_opportunity_preserved"])
        self.assertEqual(by_tag["A"]["coverage_outcome"], "coverage_preserved")
        self.assertFalse(by_tag["B"]["gt_opportunity_exists"])
        self.assertEqual(by_tag["B"]["coverage_outcome"], "coverage_preserved")

        self.assertEqual(summary[0]["n_gt_covered"], 1)
        self.assertEqual(summary[0]["n_gt_uncovered"], 1)
        self.assertEqual(summary[0]["n_sampling_failures"], 0)
        self.assertEqual(summary[0]["n_classifier_covered"], 2)
        self.assertEqual(summary[0]["n_coverage_preserved"], 2)
        self.assertEqual(summary[0]["n_selector_side_coverage_losses"], 0)

    def test_false_positive_acceptance_does_not_count_as_gt_retention(self):
        indexes, _ = oracle_coverage.audit_and_load_dataset(self.root, ["A", "B"])

        summary, rows = oracle_coverage.analyze_oracle_coverage(
            indexes, [2], self._selector_template(a_index="1")
        )
        row_a = next(row for row in rows if row["passage_id"] == "A")

        self.assertEqual(row_a["n_classifier_accepted_events"], 1)
        self.assertEqual(row_a["n_human_suited_events_preserved"], 0)
        self.assertFalse(row_a["gt_suited_opportunity_preserved"])
        self.assertEqual(row_a["coverage_outcome"], "coverage_preserved")
        self.assertTrue(row_a["coverage_preserved_by_false_positive"])
        self.assertEqual(summary[0]["n_classifier_covered"], 2)
        self.assertEqual(summary[0]["n_selector_side_coverage_losses"], 0)
        self.assertEqual(summary[0]["n_coverage_preserved_by_false_positive"], 2)

    def test_zero_inference_is_decomposed_by_gt_opportunity(self):
        indexes, _ = oracle_coverage.audit_and_load_dataset(self.root, ["A", "B"])

        summary, rows = oracle_coverage.analyze_oracle_coverage(
            indexes, [2], self._selector_template(a_index=None, b_index=None)
        )
        by_tag = {row["passage_id"]: row for row in rows}

        self.assertEqual(
            by_tag["A"]["coverage_outcome"], "selector_side_coverage_loss"
        )
        self.assertEqual(by_tag["B"]["coverage_outcome"], "sampling_failure")
        self.assertEqual(summary[0]["n_sampling_failures"], 1)
        self.assertEqual(summary[0]["n_selector_side_coverage_losses"], 1)

    def test_uuid_img_keys_keep_coverage_but_disable_event_retention(self):
        indexes, _ = oracle_coverage.audit_and_load_dataset(self.root, ["A", "B"])

        summary, rows = oracle_coverage.analyze_oracle_coverage(
            indexes, [2], self._selector_template(uuid_keys=True)
        )
        row_a = next(row for row in rows if row["passage_id"] == "A")

        self.assertTrue(row_a["classifier_covered"])
        self.assertEqual(row_a["coverage_outcome"], "coverage_preserved")
        self.assertFalse(row_a["selector_event_mapping_available"])
        self.assertIsNone(row_a["gt_suited_opportunity_preserved"])
        self.assertIsNone(summary[0]["n_gt_suited_opportunity_preserved"])

    def test_repeated_capture_events_do_not_inflate_unique_sources(self):
        indexes, _ = oracle_coverage.audit_and_load_dataset(self.root, ["A", "B"])

        _, rows = oracle_coverage.analyze_oracle_coverage(indexes, [4], None)
        row_a = next(row for row in rows if row["passage_id"] == "A")

        self.assertEqual(row_a["n_capture_events"], 5)
        self.assertEqual(row_a["n_unique_source_frames"], 3)
        self.assertEqual(row_a["n_human_suited_capture_events"], 2)
        self.assertEqual(row_a["n_available_human_suited_source_frames"], 1)
        self.assertEqual(
            row_a["n_captured_unique_human_suited_source_frames"], 1
        )
        self.assertEqual(row_a["unique_gt_suited_retention"], 1.0)
        self.assertIsNone(row_a["gt_suited_opportunity_preserved"])
        self.assertIsNone(row_a["coverage_outcome"])

    def test_observed_counts_must_match_deterministic_schedule(self):
        indexes, _ = oracle_coverage.audit_and_load_dataset(self.root, ["A", "B"])

        with self.assertRaisesRegex(ValueError, "count de captura divergente"):
            oracle_coverage.analyze_oracle_coverage(
                indexes, [2], self._selector_template(mismatched_total=True)
            )

    def test_unknown_or_missing_labels_are_rejected(self):
        index_path = self.root / "animal-tags" / "A" / "simulation_index.json"
        index = json.loads(index_path.read_text())
        index[0]["label"] = "partial"
        index_path.write_text(json.dumps(index))

        with self.assertRaisesRegex(ValueError, "label desconhecido"):
            oracle_coverage.audit_and_load_dataset(self.root, ["A", "B"])

    def test_article_cohort_and_zero_inference_golden_regression(self):
        cohort = oracle_coverage.load_cohort(
            oracle_coverage.DEFAULT_COHORT_METRICS
        )
        indexes, audit = oracle_coverage.audit_and_load_dataset(
            oracle_coverage.REPO_ROOT / "data" / "exp1", cohort
        )

        oracle_coverage.validate_article_cohort(audit)
        summaries, _ = oracle_coverage.analyze_oracle_coverage(
            indexes,
            [1, 2, 3, 4],
            oracle_coverage.DEFAULT_SELECTOR_METRICS,
        )
        oracle_coverage.validate_article_golden(summaries)


if __name__ == "__main__":
    unittest.main()
